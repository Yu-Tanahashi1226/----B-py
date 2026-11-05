"""Recognition-only fallback for known numeric ROIs; never infer a digit from its shape."""
def numeric_recognition(engine, image, minimum_confidence=.8):
    import cv2
    import numpy as np
    from board_adapter import parse_score
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ys,xs=np.where(gray>127)
    variants=[('prepared',image)]
    if len(xs):
        tight=image[max(0,int(ys.min())-8):min(len(gray),int(ys.max())+9),
                    max(0,int(xs.min())-8):min(gray.shape[1],int(xs.max())+9)]
        variants.append(('tight',tight))
    recognizer=engine.paddlex_pipeline.text_rec_model
    predictions=[]
    for name,crop in variants:
        prediction=list(recognizer([crop]))[0]
        text=str(prediction['rec_text']); confidence=float(prediction['rec_score'])
        predictions.append(dict(variant=name,text=text,confidence=confidence,value=parse_score(text)))
    accepted=[p for p in predictions if p['confidence']>=minimum_confidence and p['value'] is not None]
    values={p['value'] for p in accepted}
    # Require both views to agree; an empty detector result is never treated as zero.
    value=next(iter(values)) if len(accepted)==len(variants) and len(values)==1 else None
    return dict(method='PP-OCRv6 recognition on numeric ROI',minimum_confidence=minimum_confidence,
                predictions=predictions,accepted_value=value)
