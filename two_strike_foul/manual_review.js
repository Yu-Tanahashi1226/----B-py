/* Standalone browser review. Predictions never become labels automatically. */
'use strict';
function parseCsv(text) {
  text = text.replace(/^\uFEFF/, '');
  const rows = []; let row = [], cell = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      if (quoted && text[i + 1] === '"') { cell += '"'; i++; }
      else quoted = !quoted;
    } else if (!quoted && (c === ',' || c === '\n' || c === '\r')) {
      row.push(cell); cell = '';
      if (c !== ',') {
        if (c === '\r' && text[i + 1] === '\n') i++;
        if (row.some(v => v !== '')) rows.push(row);
        row = [];
      }
    } else cell += c;
  }
  if (quoted) throw new Error('CSVの引用符が閉じていません');
  if (cell !== '' || row.length) { row.push(cell); rows.push(row); }
  if (!rows.length) throw new Error('CSVが空です');
  const headers = rows.shift();
  if (new Set(headers).size !== headers.length) throw new Error('CSVの列名が重複しています');
  return rows.map(cells => {
    if (cells.length !== headers.length) throw new Error('CSVの列数が一致しません');
    return Object.fromEntries(headers.map((key, i) => [key, cells[i]]));
  });
}
function csvText(columns, rows) {
  const cell = v => '"' + String(v === null || v === undefined ? '' : v).replace(/"/g, '""') + '"';
  return '\uFEFF' + [columns, ...rows.map(r => columns.map(k => r[k]))].map(r => r.map(cell).join(',')).join('\r\n') + '\r\n';
}
function labelValue(field, value) {
  value = String(value).trim();
  if (!value) return '';
  if (field.startsWith('runner_') || ['board_present', 'pitch_present', 'foul', 'two_strike_foul'].includes(field)) {
    const parsed = {true:'1', false:'0', '1':'1', '0':'0'}[value.toLowerCase()];
    if (parsed === undefined) throw new Error('ありは1、なしは0を入力してください');
    return parsed;
  }
  if (field === 'half') {
    const parsed = {top:'top', bottom:'bottom', '表':'top', '裏':'bottom'}[value];
    if (!parsed) throw new Error('表・裏を選択してください');
    return parsed;
  }
  if (field.endsWith('_team')) return value;
  const n = Number(value), minimum = field === 'inning' ? 1 : 0;
  if (!Number.isFinite(n) || n < minimum || (!field.endsWith('_sec') && !Number.isInteger(n))) throw new Error('値が範囲・形式に合いません');
  if (Object.hasOwn({balls:3, strikes:2, outs:2}, field) && n > {balls:3, strikes:2, outs:2}[field]) throw new Error('カウントの上限を超えています');
  return String(n);
}
if (typeof module !== 'undefined') module.exports = {parseCsv, csvText, labelValue};
if (typeof document !== 'undefined') {
  const D = JSON.parse(document.getElementById('review-data').textContent);
  const $ = id => document.getElementById(id);
  const fields = D.fields, byFrame = new Map(D.observations.map((row, i) => [String(row[0]), i]));
  const byEvent = new Map(D.events.map(e => [e.id, e]));
  let state = structuredClone(D.state_labels), pitches = structuredClone(D.event_labels);
  const storageKey = 'baseball-review-v1:' + D.source.sha256 + ':' + D.observations_sha256;
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
  const display = value => value === null || value === undefined ? '未出力' : value === true ? 'あり' : value === false ? 'なし' : value === 'top' ? '表' : value === 'bottom' ? '裏' : typeof value === 'number' && !Number.isInteger(value) ? value.toFixed(3) : String(value);
  const status = {held:'保持', unknown:'不明', cached:'OCRキャッシュ', observed:'今回の読取り'};
  function countLabels() { return [...Object.values(state), ...Object.values(pitches)].reduce((n, r) => n + Object.keys(r).filter(k => k.startsWith('true_') && r[k] !== '').length, 0); }
  function persist() {
    try {
      localStorage.setItem(storageKey, JSON.stringify({state, pitches}));
      $('save-status').textContent = '目視入力 ' + countLabels() + ' 項目をブラウザーに保存しました。CSV保存ボタンでファイルにも保存してください。';
    } catch (_) { $('save-status').textContent = 'このブラウザーでは自動保存できません。CSV保存ボタンで入力結果を保存してください。'; }
  }
  function put(scope, key, field, value) {
    const target = scope === 'state' ? state : pitches;
    target[key] ??= {};
    if (value === '') delete target[key][field]; else target[key][field] = value;
  }
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || 'null');
    if (saved && saved.state && saved.pitches) {
      // Saved edits belong to this exact source and observation set.
      for (const [key, row] of Object.entries(saved.state)) if (byFrame.has(key)) state[key] = row;
      for (const [key, row] of Object.entries(saved.pitches)) if (byEvent.has(key)) pitches[key] = row;
    }
  } catch (_) { /* The CSV import remains available if local storage is unavailable. */ }
  function input(scope, key, field) {
    const value = (scope === 'state' ? state : pitches)[key]?.['true_' + field] ?? '';
    const attrs = `data-scope="${scope}" data-key="${esc(key)}" data-field="${field}" aria-label="目視正解 ${esc(D.labels[field] || D.model_labels[field])}"`;
    let choices = null;
    if (field.startsWith('runner_') || ['board_present', 'pitch_present', 'foul', 'two_strike_foul'].includes(field)) choices = [['','未確認'],['1','あり／はい'],['0','なし／いいえ']];
    else if (field === 'half') choices = [['','未確認'],['top','表'],['bottom','裏']];
    if (choices) return `<select ${attrs}>` + choices.map(([v,t]) => `<option value="${v}" ${String(value) === v ? 'selected' : ''}>${t}</option>`).join('') + '</select>';
    const text = field.endsWith('_team');
    return `<input ${attrs} type="${text ? 'text' : 'number'}" value="${esc(value)}" ${text ? '' : `min="${field === 'inning' ? 1 : 0}" step="${field.endsWith('_sec') ? '0.001' : '1'}" ${['balls','strikes','outs'].includes(field) ? `max="${{balls:3,strikes:2,outs:2}[field]}"` : ''}`}>`;
  }
  function note(scope, key) {
    return `<input class="field-note" type="text" data-scope="${scope}" data-key="${esc(key)}" data-field="note" value="${esc((scope === 'state' ? state : pitches)[key]?.note || '')}" aria-label="目視メモ">`;
  }
  function observedCells(row, field) {
    const i = fields.indexOf(field), text = row[5]?.[field];
    return `<td>${esc(display(row[2][i]))}${text !== null && text !== undefined ? `<small>OCR文字：${esc(text)}</small>` : ''}</td>` +
      `<td>${esc(display(row[3][i]))}<small class="${row[4][i] === 'held' ? 'held' : ''}">${esc(status[row[4][i]] || row[4][i])}</small></td>` +
      `<td class="manual">${input('state', String(row[0]), field)}</td>`;
  }
  const seekButton = (seconds, label) => `<button type="button" data-seek="${esc(seconds)}">${esc(label)}</button>`;
  let page = 0, selectedRows = D.observations.map((_, i) => i);
  function renderStates() {
    const group = $('field-group').value;
    const chosen = fields.filter(f => group === 'all' || (group === 'counts' && ['balls','strikes','outs'].includes(f)) || (group === 'runners' && (f.startsWith('runner_') || f === 'board_present')) || (group === 'text' && ['inning','half','upper_team','lower_team','upper_score','lower_score'].includes(f)));
    let text = '<thead><tr><th>フレーム／秒</th>' + chosen.map(f => `<th>${esc(D.labels[f])}<br>読取り</th><th>補正後</th><th class="manual">目視正解</th>`).join('') + '<th class="manual">目視メモ</th></tr></thead><tbody>';
    for (const index of selectedRows.slice(page*50, (page+1)*50)) {
      const row = D.observations[index];
      text += `<tr><td>${row[0]} / ${Number(row[1]).toFixed(3)}秒<br>${seekButton(row[1], 'この時刻へ')}</td>` + chosen.map(f => observedCells(row, f)).join('') + `<td class="manual">${note('state', String(row[0]))}</td></tr>`;
    }
    $('state-table').innerHTML = text + '</tbody>';
    $('page-label').textContent = selectedRows.length ? `${page+1} / ${Math.ceil(selectedRows.length/50)}ページ（${selectedRows.length}観測）` : 'この範囲には観測がありません';
    $('previous').disabled = page === 0; $('next').disabled = (page+1)*50 >= selectedRows.length;
  }
  function renderPitches() {
    if (!D.events.length) { $('pitch-list').textContent = 'VideoMAEの完了した投球結果がまだありません。全時刻表にはYOLO・OCR結果を表示できます。解析完了後にこの確認表を再作成してください。'; return; }
    $('pitch-list').innerHTML = D.events.map(e => {
      let text = `<article><h3>${esc(e.reference || '投球候補')} / ${esc(e.pitch_index || e.id)}球目</h3><p>${esc(e.status)}　投球スコア：${esc(display(e.pitch_score))}　試作ファウルスコア：${esc(e.foul_score === null ? '未実行・未出力' : display(e.foul_score))}</p>`;
      if (e.start !== null && e.start !== undefined) text += seekButton(Math.max(0,e.start-2), '投球の前から確認');
      const snapshots = ['before','after'].map(side => e.snapshots[side] === null ? null : D.observations[e.snapshots[side]]);
      text += '<div class="scroll"><table><thead><tr><th>項目</th>';
      snapshots.forEach((row, i) => { text += `<th>${i ? '後' : '前'}：読取り${row ? `<br>${Number(row[1]).toFixed(3)}秒／frame ${row[0]}` : '（時刻不明）'}</th><th>補正後</th><th class="manual">目視正解</th>`; });
      text += '</tr></thead><tbody>';
      fields.forEach(f => { text += `<tr><th>${esc(D.labels[f])}</th>` + snapshots.map(row => row ? observedCells(row, f) : '<td colspan="3">代表フレームなし。全時刻表で確認してください。</td>').join('') + '</tr>'; });
      text += '</tbody></table></div><p>';
      snapshots.forEach((row,i) => { if (row) text += seekButton(row[1], (i ? '後' : '前') + 'のスコアボードを確認'); });
      text += '</p><table class="model-table"><tr><th>VideoMAE・統合判定の項目</th><th>検出・推定</th><th class="manual">目視正解</th></tr>';
      D.model_fields.forEach(f => { text += `<tr><th>${esc(D.model_labels[f])}</th><td>${esc(display(e.predictions[f]))}</td><td class="manual">${input('event',e.id,f)}${f === 'two_strike_foul' ? '<small>実際に2ストライク後のファウルだったか</small>' : ''}</td></tr>`; });
      return text + `<tr><th>目視メモ</th><td></td><td class="manual">${note('event',e.id)}</td></tr></table></article>`;
    }).join('');
  }
  function refresh() { renderStates(); renderPitches(); }
  function updateInput(event) {
    const el = event.target, {scope, key, field} = el.dataset;
    // CSV import rebuilds the tables. A delayed blur/change from a detached old
    // control must not erase the label just restored from the imported CSV.
    if (!scope || !el.isConnected) return;
    try {
      const value = field === 'note' ? el.value : labelValue(field, el.value);
      el.setAttribute('aria-invalid','false'); el.setCustomValidity('');
      put(scope, key, field === 'note' ? 'note' : 'true_' + field, value);
      // The same source frame can appear beside more than one pitch.
      document.querySelectorAll('[data-scope]').forEach(other => {
        if (other !== el && other.dataset.scope === scope && other.dataset.key === key && other.dataset.field === field) {
          other.value = value; other.setAttribute('aria-invalid','false'); other.setCustomValidity('');
        }
      });
      persist();
    } catch (error) { el.setAttribute('aria-invalid','true'); el.setCustomValidity(error.message); $('save-status').textContent = error.message; }
  }
  // Text, number and select edits all emit input in supported browsers. A later
  // blur/change may belong to an old table, so it must not recommit stale text.
  document.addEventListener('input', updateInput);
  function stateExport() {
    return D.observations.map(row => {
      const record = {video:D.source.video, source_sha256:D.source.sha256, frame:row[0], time_sec:row[1]};
      fields.forEach((f, i) => Object.assign(record, {['pred_'+f]:row[2][i], ['recorded_'+f]:row[3][i], ['true_'+f]:state[String(row[0])]?.['true_'+f] ?? '', [f+'_status']:row[4][i]}));
      record.note = state[String(row[0])]?.note || ''; return record;
    });
  }
  function pitchExport() {
    return D.events.map((e, i) => {
      const row = {...D.pitch_templates[i]};
      ['before','after'].forEach(side => {
        const index = e.snapshots[side];
        fields.forEach(f => { row['true_'+side+'_'+f] = index === null ? '' : state[String(D.observations[index][0])]?.['true_'+f] ?? ''; });
      });
      D.model_fields.forEach(f => { row['true_'+f] = pitches[e.id]?.['true_'+f] ?? ''; });
      row.note = pitches[e.id]?.note || ''; return row;
    });
  }
  function download(name, columns, rows) {
    if (document.querySelector('[aria-invalid=true]')) { $('save-status').textContent = '赤枠の入力を修正してからCSVを保存してください。'; return; }
    for (const record of pitchExport()) {
      if (record.true_pitch_start_sec !== '' && record.true_pitch_end_sec !== '' && Number(record.true_pitch_start_sec) >= Number(record.true_pitch_end_sec)) {
        $('save-status').textContent = '目視の投球開始秒を終了秒より前にしてください。'; return;
      }
    }
    const blob = new Blob([csvText(columns, rows)], {type:'text/csv;charset=utf-8'}), url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href=url; a.download=name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 10000);
    $('save-status').textContent = name + ' の保存を開始しました。ブラウザーのダウンロード先を確認してください。';
  }
  $('save-states').onclick = () => download('scoreboard_review.csv',D.state_columns,stateExport());
  $('save-pitches').onclick = () => download('pitch_review.csv',D.pitch_columns,pitchExport());
  $('import-csv').onchange = async event => {
    try {
      const rows = parseCsv(await event.target.files[0].text()), changes = [];
      for (const row of rows) {
        if (row.source_sha256 !== D.source.sha256 || row.video !== D.source.video) throw new Error('CSVの動画名・ハッシュが一致しません');
        const isState = Object.hasOwn(row, 'frame'), key = isState ? String(Number(row.frame)) : String(row.event_id), e = isState ? null : byEvent.get(key);
        if (isState ? !byFrame.has(key) : !e) throw new Error('CSVにこの確認表にないフレーム・投球があります');
        if (e && row.reference && row.reference !== e.reference) throw new Error('投球番号と参考動画が一致しません');
        if (isState) {
          fields.forEach(f => { if (Object.hasOwn(row,'true_'+f)) changes.push(['state',key,'true_'+f,labelValue(f,row['true_'+f])]); });
        } else {
          D.model_fields.forEach(f => { if (Object.hasOwn(row,'true_'+f)) changes.push(['event',key,'true_'+f,labelValue(f,row['true_'+f])]); });
          ['before','after'].forEach(side => {
            if (e.snapshots[side] === null) return;
            const frame = String(D.observations[e.snapshots[side]][0]);
            fields.forEach(f => { if (Object.hasOwn(row,'true_'+side+'_'+f)) changes.push(['state',frame,'true_'+f,labelValue(f,row['true_'+side+'_'+f])]); });
          });
        }
        if (Object.hasOwn(row,'note')) changes.push([isState ? 'state' : 'event',key,'note',row.note]);
      }
      const pending = new Map();
      for (const change of changes) { const k=JSON.stringify(change.slice(0,3)); if (pending.has(k) && pending.get(k) !== change[3]) throw new Error('CSV内の同じ正解項目に異なる値があります'); pending.set(k,change[3]); }
      changes.forEach(c => put(...c)); persist(); refresh(); $('import-status').textContent = '目視入力を読み込みました。';
    } catch (error) { $('import-status').textContent = '読み込めませんでした：' + error.message; }
    event.target.value = '';
  };
  let videoUrl = null, pendingSeek = null;
  $('video-file').onchange = event => {
    const file = event.target.files[0]; if (!file) return;
    if (file.name !== D.source.video) { $('video-status').textContent = '元動画 ' + D.source.video + ' を選んでください。'; return; }
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    videoUrl = URL.createObjectURL(file); $('video').src=videoUrl;
    $('video-status').textContent = file.name + ' を読み込みました。';
  };
  $('video').onloadedmetadata = () => { if (pendingSeek !== null) { $('video').currentTime=pendingSeek; pendingSeek=null; } };
  document.addEventListener('click', event => {
    const button = event.target.closest('button[data-seek]'); if (!button) return;
    const t=Number(button.dataset.seek);
    if (!$('video').src) { $('video-status').textContent = '先に元動画 ' + D.source.video + ' を選んでください。'; $('video-file').scrollIntoView({behavior:'smooth'}); return; }
    if ($('video').readyState) $('video').currentTime=Math.min(t,$('video').duration); else pendingSeek=t;
    $('video').pause(); $('video').scrollIntoView({behavior:'smooth'});
  });
  function tab(which) {
    $('states').hidden=which!=='states'; $('pitches').hidden=which!=='pitches';
    $('show-states').setAttribute('aria-pressed',String(which==='states')); $('show-pitches').setAttribute('aria-pressed',String(which==='pitches'));
  }
  $('show-states').onclick=()=>tab('states'); $('show-pitches').onclick=()=>tab('pitches');
  $('field-group').onchange=renderStates;
  $('apply-range').onclick=()=>{const a=$('from-sec').value==='' ? -Infinity : Number($('from-sec').value), b=$('to-sec').value==='' ? Infinity : Number($('to-sec').value);selectedRows=D.observations.flatMap((r,i)=>r[1]>=a && r[1]<=b ? [i] : []);page=0;renderStates();};
  $('previous').onclick=()=>{page--;renderStates();}; $('next').onclick=()=>{page++;renderStates();};
  $('source').textContent=D.source.video + ' ／ ' + D.observations.length + '観測・' + D.events.length + '投球結果';
  $('save-pitches').disabled=!D.events.length;
  refresh(); tab(D.events.length ? 'pitches' : 'states');
  $('save-status').textContent='目視入力 ' + countLabels() + ' 項目。入力後はCSV保存ボタンでファイルへ保存してください。';
}
