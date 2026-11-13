// Optional UI verification; run with Playwright + Chromium in a test environment.
// Labels entered here are synthetic test inputs, not research ground truth.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const http = require('node:http');
const {chromium} = require('playwright');
const {parseCsv, csvText, labelValue} = require('../manual_review.js');

async function main() {
  const [pagePath, output] = process.argv.slice(2);
  assert(pagePath && output, 'Usage: node manual_review_browser.cjs PAGE.html OUTPUT_DIR');
  await fs.mkdir(output, {recursive:true});
  assert.deepEqual(parseCsv(csvText(['a','b'], [{a:'確認, "済"\n次の行',b:'0'}])), [{a:'確認, "済"\n次の行',b:'0'}]);
  assert.equal(labelValue('runner_first','false'),'0');
  assert.throws(()=>labelValue('balls','4'));
  const server=http.createServer(async (_request,response)=>{response.setHeader('Content-Type','text/html; charset=utf-8');response.end(await fs.readFile(pagePath));});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const browser = await chromium.launch({executablePath:process.env.CHROMIUM_PATH || '/usr/bin/chromium', headless:true, args:['--no-sandbox']});
  const context = await browser.newContext({acceptDownloads:true,viewport:{width:1450,height:1000}});
  const page = await context.newPage(); const errors=[];
  await page.addInitScript(()=>{
    window.reviewTestEvents=[];
    const write=Storage.prototype.setItem;
    Storage.prototype.setItem=function(k,v){try{const d=JSON.parse(v);window.reviewTestEvents.push({type:'save',balls:Object.entries(d.state||{}).filter(([_,r])=>r.true_balls)});}catch{}return write.call(this,k,v);};
    for(const type of ['input','change'])document.addEventListener(type,e=>window.reviewTestEvents.push({type,id:e.target.id,key:e.target.dataset?.key,field:e.target.dataset?.field,value:e.target.value}),true);
  });
  page.on('pageerror', e => errors.push(String(e)));
  try {
    await page.goto('http://127.0.0.1:'+server.address().port+'/index.html');
    const data=await page.locator('#review-data').evaluate(el=>JSON.parse(el.textContent));
    assert(data.events.length>0);
    await page.locator('#pitch-list article').first().scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(output,'initial_blank_review.png'),fullPage:false});
    const event=data.events[0], frame=String(data.observations[event.snapshots.before][0]);
    const countSelector=`[data-scope="state"][data-key="${frame}"][data-field="balls"]`;
    await page.locator(countSelector).first().fill('1');
    await page.locator(`[data-scope="event"][data-key="${event.id}"][data-field="foul"]`).selectOption('1');
    await page.locator(`[data-scope="event"][data-key="${event.id}"][data-field="note"]`).fill('UI試験用, "正解ではない"');
    let waiting=page.waitForEvent('download');await page.locator('#save-states').click();
    const stateDownload=await waiting, statePath=path.join(output,'synthetic_state_labels.csv');
    await stateDownload.saveAs(statePath);
    const states=parseCsv(await fs.readFile(statePath,'utf8'));
    assert.equal(states.length,data.observations.length);
    assert.equal(states.find(r=>r.frame===frame).true_balls,'1');
    assert(states.every(r=>r.true_strikes===''));
    waiting=page.waitForEvent('download');await page.locator('#save-pitches').click();
    const pitchDownload=await waiting,pitchPath=path.join(output,'synthetic_pitch_labels.csv');
    await pitchDownload.saveAs(pitchPath);
    const pitches=parseCsv(await fs.readFile(pitchPath,'utf8'));
    assert.equal(pitches[0].true_before_balls,'1');
    assert.equal(pitches[0].true_foul,'1');
    assert.equal(pitches[0].note,'UI試験用, "正解ではない"');
    assert.equal(pitches[0].true_pitch_present,'');
    await page.locator(countSelector).first().fill('');
    await page.locator('#import-csv').setInputFiles(statePath);
    await page.waitForFunction(()=>document.getElementById('import-status').textContent.includes('読み込みました'));
    assert.equal(await page.locator(countSelector).first().inputValue(),'1');
    const storageStatus=await page.locator('#save-status').textContent();
    const beforeReload=await page.evaluate(()=>({keys:Object.keys(localStorage),values:Object.keys(localStorage).map(k=>[k,localStorage.getItem(k)]),status:document.getElementById('save-status').textContent,trace:window.reviewTestEvents}));
    if (storageStatus.includes('自動保存できません')) {
      // Some managed browser policies disable localStorage. CSV import is the
      // required durable fallback in that environment.
      await page.reload();await page.locator('#import-csv').setInputFiles(statePath);
      await page.waitForFunction(()=>document.getElementById('import-status').textContent.includes('読み込みました'));
    } else { await page.reload(); }
    if (await page.locator(countSelector).first().inputValue() !== '1') {
      const afterReload=await page.evaluate(()=>({keys:Object.keys(localStorage),values:Object.keys(localStorage).map(k=>[k,localStorage.getItem(k)]),status:document.getElementById('save-status').textContent}));
      await fs.writeFile(path.join(output,'storage_debug.json'),JSON.stringify({beforeReload,afterReload,errors},null,2));
    }
    assert.equal(await page.locator(countSelector).first().inputValue(),'1');
    const wrongPath=path.join(output,'wrong_source.csv');
    await fs.writeFile(wrongPath,csvText(['video','source_sha256','frame','true_balls'],[{video:data.source.video,source_sha256:'wrong',frame,true_balls:'3'}]));
    await page.locator('#import-csv').setInputFiles(wrongPath);
    await page.waitForFunction(()=>document.getElementById('import-status').textContent.includes('一致しません'));
    assert.equal(await page.locator(countSelector).first().inputValue(),'1');
    await page.locator('#show-states').click();
    assert.equal(await page.locator('#state-table tbody tr').count(),Math.min(50,data.observations.length));
    await page.locator('#field-group').selectOption('text');
    assert((await page.locator('#state-table thead').textContent()).includes('チーム'));
    await page.locator('#from-sec').fill('105');await page.locator('#to-sec').fill('120');await page.locator('#apply-range').click();
    assert((await page.locator('#page-label').textContent()).includes('観測'));
    assert.deepEqual(errors,[]);
    await fs.writeFile(path.join(output,'browser_verification.json'),JSON.stringify({passed:true,observations:data.observations.length,events:data.events.length,automatic_browser_save:!storageStatus.includes('自動保存できません'),tests:['editable paired inputs','CSV export','CSV import','source identity rejection','reload recovery','frame/pitch label synchronization','range and field filters'],synthetic_labels_not_ground_truth:true},null,2));
    console.log('Browser review verified:',data.observations.length,'observations,',data.events.length,'pitches');
  } finally {await browser.close();await new Promise(resolve=>server.close(resolve));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
