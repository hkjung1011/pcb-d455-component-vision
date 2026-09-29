from pathlib import Path
import csv
import hashlib
import json
import shutil

TRIAL=Path(__file__).resolve().parent/'boards8_pilot_20260929'
RUN=TRIAL/'runs/boards8_yolo11s_10ep_attempt2'
STATE=TRIAL/'execution_state_attempt2.json'
SUMMARY=RUN/'training-summary.json'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
state=read(STATE)
summary=read(SUMMARY)
assert state['status']=='complete'
epoch_numbers={int(row['epoch']) for row in csv.DictReader((RUN/'results.csv').open(encoding='utf-8-sig'))}
assert epoch_numbers==set(range(1,11))
extra=[row for row in state['epochs'] if row['epoch'] not in epoch_numbers]
assert len(extra)==1 and extra[0]['epoch']==11
assert extra[0]['optimizer_steps']==state['epochs'][-2]['optimizer_steps']==235
checkpoint_before=sha(Path(summary['best']))
test_before=sha(RUN/'test-evaluation.json')
shutil.copyfile(STATE,TRIAL/'execution_state_before_epoch_accounting_correction.json')
shutil.copyfile(SUMMARY,RUN/'training-summary-before-epoch-accounting-correction.json')
state['epochs']=[r for r in state['epochs'] if r['epoch'] in epoch_numbers]
state['post_training_validation_callbacks']=extra
state['epoch_accounting_note']='Ultralytics final_eval repeats on_fit_epoch_end after advancing trainer.epoch. Actual training is the 10 CSV epoch rows; extra callback had no optimizer updates.'
summary['epochs_completed']=len(epoch_numbers)
summary['epoch_accounting_note']=state['epoch_accounting_note']
write(STATE,state)
write(SUMMARY,summary)
correction={'scope':'Training epoch accounting only','actual_training_epochs':10,'extra_validation_callbacks':1,'optimizer_steps':235,'checkpoint_sha256':checkpoint_before,'checkpoint_unchanged':checkpoint_before==sha(Path(summary['best'])),'test_metrics_sha256':test_before,'test_metrics_unchanged':test_before==sha(RUN/'test-evaluation.json'),'test_repeated':False,'source_fix':'Ignore callbacks beyond configured epochs and derive completed epoch count from results.csv'}
write(RUN/'epoch-accounting-correction.json',correction)
print(json.dumps(correction,indent=2))
