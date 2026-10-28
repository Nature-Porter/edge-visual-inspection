"""Validate a complete C++ replay against event schema and summary counts."""
import argparse
import json
from collections import Counter
from jsonschema import Draft202012Validator
from common import ROOT,read,save,sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run=ROOT/a.run;events_path=run/'events.jsonl';summary_path=run/'summary.json'
    schema_path=ROOT/'schemas/event.schema.json';schema=read(schema_path)
    validator=Draft202012Validator(schema)
    snapshot=read(run/'snapshot.json');summary=read(summary_path)
    expected_model=snapshot['manifest']['onnx_sha256'];allowed_sources={r['source_image_id'] for r in snapshot['source']}
    ids=set();statuses=Counter();decisions=Counter();errors=[]
    with events_path.open(encoding='utf-8') as stream:
        for line_number,line in enumerate(stream,1):
            try:
                event=json.loads(line)
                issue=next(validator.iter_errors(event),None)
                if issue:raise ValueError(issue.message)
                frame_id=event['frame_id']
                if frame_id in ids:raise ValueError('duplicate frame_id')
                ids.add(frame_id)
                if event['model_sha256']!=expected_model:raise ValueError('model hash mismatch')
                if event['source_image_id'] not in allowed_sources:raise ValueError('source outside replay')
                statuses[event['status']]+=1;decisions[event['decision']]+=1
            except (ValueError,json.JSONDecodeError) as exc:
                errors.append({'line':line_number,'reason':str(exc)})
                if len(errors)>=20:break
    if ids!=set(range(1,summary['generated']+1)):errors.append({'reason':'missing or out-of-range frame IDs'})
    if len(ids)!=summary['completed']:errors.append({'reason':'summary completed mismatch'})
    if statuses['QUEUE_OVERFLOW']!=summary['queue_overflow'] or statuses['TIMEOUT']!=summary['timeout'] or decisions['REVIEW']!=summary['review']:
        errors.append({'reason':'summary status/decision counts mismatch'})
    result={'status':'passed' if not errors else 'failed','run':a.run,'event_count':len(ids),'statuses':dict(statuses),'decisions':dict(decisions),
        'events_sha256':sha256(events_path),'summary_sha256':sha256(summary_path),'schema_sha256':sha256(schema_path),'errors':errors}
    save(ROOT/a.output,result);print(result['status'],len(ids));raise SystemExit(0 if not errors else 5)

if __name__=='__main__':main()
