########################################################
# 2026.10.28, current-execution metrics / 不将旧模型或失败 section 的旧指标混入新表
########################################################
from datetime import datetime,timezone
from pathlib import Path
import json
import os
import tempfile
import uuid
import pandas as pd
from pipeline_provenance import fingerprint,sha256,atomic_json


def atomic_csv(path,frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent);os.close(fd)
    try:frame.to_csv(name,index=False);os.replace(name,path)
    finally:Path(name).unlink(missing_ok=True)


class EvaluationRun:
    def __init__(self,root,requested,*,context):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_'+uuid.uuid4().hex[:8]
        self.history=self.root/'evaluation_history'/self.id;self.history.mkdir(parents=True)
        self.requested={str(x['section_id']):dict(x) for x in requested}
        if len(self.requested)!=len(requested):raise ValueError('Duplicate section ID')
        self.context=context;self.rows={};self.status={key:{'status':'pending'} for key in self.requested}
        old=self.root/'metrics_per_section.csv'
        if old.is_file():
            import shutil
            shutil.copy2(old,self.history/'previous_NOT_CURRENT.csv')
        self.publish('running')

    def success(self,sid,row):
        sid=str(sid);expected=self.requested[sid]
        for key in ('section_id','dataset','sample'):
            if str(row.get(key))!=str(expected[key]):raise ValueError('Returned section identity mismatch')
        if row.get('evaluation_scope')!=self.context['scope']:raise ValueError('Evaluation scope mismatch')
        if self.context.get('mapping_sha256')!=row.get('mapping_sha256'):raise ValueError('Mapping changed during evaluation')
        if self.context.get('model_family')!=row.get('model_family'):raise ValueError('Mixed model protocols')
        if 'native_col' in expected and str(row.get('requested_native_col'))!=str(expected['native_col']):
            raise ValueError('GT column mismatch')
        if not row.get('run_id'):raise ValueError('Missing input fingerprint')
        self.rows[sid]=dict(row,execution_id=self.id)
        self.status[sid]={'status':row.get('prediction_status','computed'),'run_id':row['run_id']}
        self.journal('running')

    def failure(self,sid,error):
        self.rows.pop(str(sid),None);self.status[str(sid)]={'status':'failed','error':str(error)}
        self.journal('running')

    def journal(self,state,csv_sha=None):
        out={'version':1,'execution_id':self.id,'state':state,'context':self.context,
            'requested':list(self.requested.values()),'sections':self.status,
            'n_success':len(self.rows),'n_requested':len(self.requested),'metrics_sha256':csv_sha}
        atomic_json(self.history/'evaluation_run.json',out);atomic_json(self.root/'evaluation_run.json',out)

    def publish(self,state):
        frame=pd.DataFrame(list(self.rows.values()))
        if frame.empty:frame=pd.DataFrame(columns=['section_id','dataset','sample','execution_id','evaluation_scope',
            'checkpoint_sha256','run_id','macro_auc_L1_cells','macro_auc_L12_cells','macro_auc_L2_cells'])
        atomic_csv(self.root/'metrics_per_section.csv',frame);atomic_csv(self.history/'metrics_per_section.csv',frame)
        self.journal(state,sha256(self.root/'metrics_per_section.csv'));return frame

    def finish(self):
        for sid,status in list(self.status.items()):
            if status['status']=='pending':self.failure(sid,'Not processed')
        state='completed_with_failures' if any(x['status']=='failed' for x in self.status.values()) else 'completed'
        return self.publish(state)
