import zipfile, re, xml.etree.ElementTree as ET, math, os, itertools, csv, json
from collections import Counter
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR='results'
os.makedirs(OUTDIR, exist_ok=True)
SEED=42
XLSX='../../data/data_qsar_pls_extended.xlsx'
NS={'a':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

def parse_xlsx(path):
    with zipfile.ZipFile(path) as z:
        ss=[]
        root=ET.fromstring(z.read('xl/sharedStrings.xml'))
        for si in root.findall('a:si', NS):
            ss.append(''.join(t.text or '' for t in si.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t')))
        root=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        def cidx(ref):
            m=re.match(r'([A-Z]+)', ref); n=0
            for ch in m.group(1): n=n*26+ord(ch)-64
            return n-1
        rows=[]
        for row in root.findall('.//a:sheetData/a:row', NS):
            vals=[]
            for c in row.findall('a:c', NS):
                j=cidx(c.attrib['r'])
                while len(vals)<j: vals.append(None)
                v=c.find('a:v', NS); t=c.attrib.get('t'); val=None
                if v is not None:
                    raw=v.text
                    if t=='s': val=ss[int(raw)]
                    else:
                        try:
                            val=float(raw)
                            if val.is_integer(): val=int(val)
                        except Exception:
                            val=raw
                vals.append(val)
            rows.append(vals)
        maxc=max(len(r) for r in rows)
        for r in rows: r.extend([None]*(maxc-len(r)))
        return rows

def clean(s): return str(s).replace('\xa0',' ').strip() if s is not None else ''
def sf(v):
    if v is None or v=='': return np.nan
    if isinstance(v,(int,float)): return float(v)
    try: return float(str(v).replace(',','.'))
    except Exception: return np.nan

def rmse(a,b): return float(np.sqrt(mean_squared_error(a,b)))
def q2(a,b):
    a=np.asarray(a); b=np.asarray(b); den=np.sum((a-a.mean())**2)
    return float(1-np.sum((a-b)**2)/den) if den else np.nan

def metrics(y,p):
    return {'R2':float(r2_score(y,p)), 'RMSE':rmse(y,p), 'MAE':float(mean_absolute_error(y,p))}

def vif(X):
    X=np.asarray(X,dtype=float)
    vals=[]
    for i in range(X.shape[1]):
        if X.shape[1]==1: vals.append(1.0); continue
        y=X[:,i]; Z=np.delete(X,i,axis=1)
        lr=LinearRegression().fit(Z,y); r2=lr.score(Z,y)
        vals.append(float('inf') if r2>=0.999999 else float(1/(1-r2)))
    return vals

def leverage_train(Xstd):
    Xd=np.c_[np.ones(Xstd.shape[0]), Xstd]
    H=Xd @ np.linalg.pinv(Xd.T@Xd) @ Xd.T
    return np.diag(H)

def leverage_new(Xstd_train, Xstd_new):
    Xd=np.c_[np.ones(Xstd_train.shape[0]), Xstd_train]
    inv=np.linalg.pinv(Xd.T@Xd)
    Xn=np.c_[np.ones(Xstd_new.shape[0]), Xstd_new]
    return np.sum((Xn@inv)*Xn,axis=1)

def min_dist(Xtrain_std, Xnew_std):
    out=[]
    for x in Xnew_std:
        out.append(float(np.min(np.sqrt(((Xtrain_std-x)**2).sum(axis=1)))))
    return np.array(out)

rows=parse_xlsx(XLSX)
headers=[clean(x) for x in rows[0]]
records=[]
for i,r in enumerate(rows[1:], start=2):
    rec={headers[j]:r[j] for j in range(len(headers))}
    rec['_row']=i; records.append(rec)
name_col='Organic compounds'; y_col='EXP LogKda'; set_col='Set'
model=[r for r in records if r.get(set_col) in ('TS','VS') and not np.isnan(sf(r.get(y_col)))]
ext=[r for r in records if r.get(set_col)=='EXT']
# Descriptor cols from file
non={name_col,y_col,set_col,'_row',''}
desc=[]
for h in headers:
    if h not in non and h not in desc: desc.append(h)
# Diagnostics
allX=np.array([[sf(r.get(c)) for c in desc] for r in model], dtype=float)
missing_cols=[(c,int(np.isnan(allX[:,i]).sum())) for i,c in enumerate(desc) if np.isnan(allX[:,i]).sum()>0]
nonnum=[]
for c in desc:
    for r in model:
        v=r.get(c)
        if v not in (None,'') and np.isnan(sf(v)):
            nonnum.append(c); break
const=[]; near_const=[]
for i,c in enumerate(desc):
    col=allX[:,i]
    col=col[~np.isnan(col)]
    if len(col)==0 or np.nanstd(col)==0: const.append(c)
    elif len(np.unique(np.round(col,10)))<=2: near_const.append(c)
# duplicates by name and by y + six candidate descriptors only (not full 2000-d)
names=[r[name_col] for r in model]
dup_names={k:v for k,v in Counter(names).items() if v>1}
# endpoint outliers via IQR
Y=np.array([sf(r[y_col]) for r in model])
q1,q3=np.percentile(Y,[25,75]); IQR=q3-q1
resp_out=[names[i] for i,y in enumerate(Y) if y<q1-1.5*IQR or y>q3+1.5*IQR]

features=['log D', "Mw'", 'εα', 'εβ', "V'", 'π']
X=np.array([[sf(r.get(f)) for f in features] for r in model], dtype=float)
sets=np.array([r[set_col] for r in model])
tr=sets=='TS'; te=sets=='VS'
Xtr_all,Xte_all=X[tr],X[te]; ytr,yte=Y[tr],Y[te]
names_tr=np.array(names)[tr].tolist(); names_te=np.array(names)[te].tolist()
# correlation candidate TS
corr=np.corrcoef(Xtr_all,rowvar=False)
# feature z outliers candidate set
z=(X-X.mean(axis=0))/X.std(axis=0)
zout=[]
for i,n in enumerate(names):
    f=[features[j] for j in range(len(features)) if abs(z[i,j])>3]
    if f: zout.append({'Compound':n,'features':'; '.join(f)})

cv=KFold(n_splits=5, shuffle=True, random_state=SEED)
# MLR subset selection (TS only)
subsets=[]
for k in range(1,5):
    for idxs in itertools.combinations(range(len(features)), k):
        Xs=Xtr_all[:,idxs]
        scaler=StandardScaler().fit(Xs)
        try:
            v=vif(scaler.transform(Xs))
        except Exception:
            continue
        maxv=max(v)
        if maxv>20: continue
        if k>1:
            c=np.corrcoef(Xs,rowvar=False)
            maxcorr=float(np.max(np.abs(c[np.triu_indices(k,1)])))
        else: maxcorr=0.0
        pipe=Pipeline([('scaler',StandardScaler()),('lr',LinearRegression())])
        cvp=cross_val_predict(pipe,Xs,ytr,cv=cv)
        pipe.fit(Xs,ytr)
        ptr=pipe.predict(Xs); pte=pipe.predict(Xte_all[:,idxs])
        subsets.append({'features':[features[i] for i in idxs], 'idxs':idxs, 'n_features':k,
                        'Q2CV':q2(ytr,cvp),'RMSECV':rmse(ytr,cvp),'MAECV':float(mean_absolute_error(ytr,cvp)),
                        'R2_TS':float(r2_score(ytr,ptr)),'RMSEC':rmse(ytr,ptr),'MAE_TS':float(mean_absolute_error(ytr,ptr)),
                        'Q2Ext_R2_VS':float(r2_score(yte,pte)),'RMSEP':rmse(yte,pte),'MAE_VS':float(mean_absolute_error(yte,pte)),
                        'maxcorr':maxcorr,'max_vif':float(maxv)})
# selection is TS CV, with VIF constraint; prefer better Q2CV, then fewer features
subsets_sorted=sorted(subsets, key=lambda x:(x['Q2CV'],-x['n_features']), reverse=True)
mlr_sel=subsets_sorted[0]
mlr_idxs=mlr_sel['idxs']; mlr_features=mlr_sel['features']
mlr=Pipeline([('scaler',StandardScaler()),('lr',LinearRegression())]).fit(Xtr_all[:,mlr_idxs],ytr)
mlr_cvp=cross_val_predict(Pipeline([('scaler',StandardScaler()),('lr',LinearRegression())]),Xtr_all[:,mlr_idxs],ytr,cv=cv)
# kNN and DT with all 6 descriptors, hyperparams by TS CV
knn_grid={'knn__n_neighbors':list(range(1,11)), 'knn__weights':['uniform','distance'], 'knn__p':[1,2]}
knn_search=GridSearchCV(Pipeline([('scaler',StandardScaler()),('knn',KNeighborsRegressor())]),knn_grid,cv=cv,scoring='neg_root_mean_squared_error')
knn_search.fit(Xtr_all,ytr); knn=knn_search.best_estimator_
knn_cvp=cross_val_predict(knn,Xtr_all,ytr,cv=cv)
dt_grid={'dt__max_depth':[1,2,3,4,5,None], 'dt__min_samples_leaf':[1,2,3,4,5], 'dt__min_samples_split':[2,4,6]}
dt_search=GridSearchCV(Pipeline([('scaler',StandardScaler()),('dt',DecisionTreeRegressor(random_state=SEED))]),dt_grid,cv=cv,scoring='neg_root_mean_squared_error')
dt_search.fit(Xtr_all,ytr); dt=dt_search.best_estimator_
dt_cvp=cross_val_predict(dt,Xtr_all,ytr,cv=cv)
models={
    'MLR': {'est':mlr,'features':mlr_features,'idxs':mlr_idxs,'Xtr':Xtr_all[:,mlr_idxs],'Xte':Xte_all[:,mlr_idxs],'cvp':mlr_cvp,'params':'subset by 5-fold CV + VIF<20'},
    'kNN': {'est':knn,'features':features,'idxs':tuple(range(6)),'Xtr':Xtr_all,'Xte':Xte_all,'cvp':knn_cvp,'params':knn_search.best_params_},
    'Decision Tree': {'est':dt,'features':features,'idxs':tuple(range(6)),'Xtr':Xtr_all,'Xte':Xte_all,'cvp':dt_cvp,'params':dt_search.best_params_}
}
summary=[]; predictions=[]
for m,info in models.items():
    est=info['est'].fit(info['Xtr'],ytr)
    ptr=est.predict(info['Xtr']); pte=est.predict(info['Xte'])
    row={'Model':m,'Descriptors':', '.join(info['features']),'n_features':len(info['features']),
         'R2_TS':float(r2_score(ytr,ptr)),'RMSEC':rmse(ytr,ptr),'MAE_TS':float(mean_absolute_error(ytr,ptr)),
         'Q2CV':q2(ytr,info['cvp']),'RMSECV':rmse(ytr,info['cvp']),'MAE_CV':float(mean_absolute_error(ytr,info['cvp'])),
         'Q2Ext_R2_VS':float(r2_score(yte,pte)),'RMSEP':rmse(yte,pte),'MAE_VS':float(mean_absolute_error(yte,pte)),
         'Overfit_gap_R2_TS_minus_Q2Ext':float(r2_score(ytr,ptr)-r2_score(yte,pte)),
         'CV_selected_params':json.dumps(info['params'],ensure_ascii=False)}
    summary.append(row)
    for n,y,p in zip(names_tr,ytr,ptr): predictions.append({'Model':m,'Set':'TS','Compound':n,'Observed_LogKd':float(y),'Predicted_LogKd':float(p),'Residual_obs_minus_pred':float(y-p)})
    for n,y,p in zip(names_te,yte,pte): predictions.append({'Model':m,'Set':'VS','Compound':n,'Observed_LogKd':float(y),'Predicted_LogKd':float(p),'Residual_obs_minus_pred':float(y-p)})
# choose best by RMSEP; if within 0.05 prefer MLR for interpretability
min_rmsep=min(r['RMSEP'] for r in summary)
cands=[r for r in summary if r['RMSEP']<=min_rmsep+0.05]
rank={'MLR':0,'kNN':1,'Decision Tree':2}
best_row=sorted(cands,key=lambda r:(rank.get(r['Model'],9), -r['Q2CV']))[0]
best_name=best_row['Model']; best_info=models[best_name]; best=best_info['est'].fit(best_info['Xtr'],ytr)
# true external supplied values. Use only descriptors selected by best model; no invented descriptors.
provided_ext={
    'Naphthalene': {'log D':3.30, "Mw'":1.281, 'εα':0.255, 'εβ':0.296, "V'":1.109, 'π':1.241},
    'Nitrobenzene': {'log D':1.85, "Mw'":1.231, 'εα':0.291, 'εβ':0.327, "V'":0.738, 'π':0.896}
}
true_ext=[]
Xext=[]
for comp,vals in provided_ext.items():
    missing=[f for f in best_info['features'] if f not in vals]
    row=[vals.get(f,np.nan) for f in best_info['features']]
    Xext.append(row)
    true_ext.append({'Compound':comp,'Set':'TRUE_EXT','Observed_LogKd':None,'Descriptors_used':', '.join(best_info['features']),'Missing_features':'; '.join(missing)})
Xext=np.array(Xext,dtype=float)
if not np.isnan(Xext).any():
    pext=best.predict(Xext)
else:
    pext=np.array([np.nan]*len(true_ext))
for r,p in zip(true_ext,pext):
    r['Predicted_LogKd']=None if np.isnan(p) else float(p)
    r['Residual_obs_minus_pred']=None
# add true external rows to prediction table for best model only
for r in true_ext:
    predictions.append({'Model':best_name,'Set':'TRUE_EXT','Compound':r['Compound'],'Observed_LogKd':None,'Predicted_LogKd':r['Predicted_LogKd'],'Residual_obs_minus_pred':None})
# applicability domain: distance for all, leverage if MLR. Since best likely MLR/knn based on data.
scaler=best.named_steps['scaler'] if hasattr(best,'named_steps') and 'scaler' in best.named_steps else StandardScaler().fit(best_info['Xtr'])
Xtr_std=scaler.transform(best_info['Xtr']); Xte_std=scaler.transform(best_info['Xte']); Xext_std=scaler.transform(Xext) if not np.isnan(Xext).any() else np.full_like(Xext,np.nan)
train_nn=[]
for i,x in enumerate(Xtr_std):
    train_nn.append(float(np.min(np.sqrt(((np.delete(Xtr_std,i,axis=0)-x)**2).sum(axis=1)))))
dist_thr=float(np.percentile(train_nn,95))
h_star=float(3*(len(best_info['features'])+1)/len(ytr))
ptr_best=best.predict(best_info['Xtr']); pte_best=best.predict(best_info['Xte'])
resid_sd=float(np.std(ytr-ptr_best,ddof=1))
lev_te=None; lev_ext=None; lev_tr=None
if best_name=='MLR':
    lev_tr=leverage_train(Xtr_std); lev_te=leverage_new(Xtr_std,Xte_std); lev_ext=leverage_new(Xtr_std,Xext_std) if not np.isnan(Xext_std).any() else None
ad=[]
for i,(comp,y,p) in enumerate(zip(names_te,yte,pte_best)):
    d=float(min_dist(Xtr_std,Xte_std[[i]])[0])
    range_ok=all(best_info['Xtr'][:,j].min()-1e-12 <= best_info['Xte'][i,j] <= best_info['Xtr'][:,j].max()+1e-12 for j in range(len(best_info['features'])))
    row={'Compound':comp,'Set':'VS','Observed_LogKd':float(y),'Predicted_LogKd':float(p),'Residual':float(y-p),
         'Nearest_distance_std':d,'Distance_threshold':dist_thr,'Distance_AD':'inside' if d<=dist_thr else 'outside','Range_AD':'inside' if range_ok else 'outside'}
    if best_name=='MLR':
        row['Leverage']=float(lev_te[i]); row['Leverage_threshold_h*']=h_star; row['Leverage_AD']='inside' if lev_te[i]<=h_star else 'outside'; row['Std_residual']=float((y-p)/resid_sd) if resid_sd else None
    row['Overall_AD']='inside' if row['Distance_AD']=='inside' and row['Range_AD']=='inside' and (best_name!='MLR' or row['Leverage_AD']=='inside') else 'outside/warning'
    ad.append(row)
# True external AD
for i,r in enumerate(true_ext):
    d=None if np.isnan(Xext_std).any() else float(min_dist(Xtr_std,Xext_std[[i]])[0])
    range_ok=not np.isnan(Xext).any()
    if range_ok:
        range_ok=all(best_info['Xtr'][:,j].min()-1e-12 <= Xext[i,j] <= best_info['Xtr'][:,j].max()+1e-12 for j in range(len(best_info['features'])))
    row={'Compound':r['Compound'],'Set':'TRUE_EXT','Observed_LogKd':None,'Predicted_LogKd':r['Predicted_LogKd'],'Residual':None,
         'Nearest_distance_std':d,'Distance_threshold':dist_thr,'Distance_AD':'unknown' if d is None else ('inside' if d<=dist_thr else 'outside'),'Range_AD':'inside' if range_ok else 'outside'}
    if best_name=='MLR':
        lv=None if lev_ext is None else float(lev_ext[i]); row['Leverage']=lv; row['Leverage_threshold_h*']=h_star; row['Leverage_AD']='unknown' if lv is None else ('inside' if lv<=h_star else 'outside'); row['Std_residual']=None
    row['Overall_AD']='inside' if row['Distance_AD']=='inside' and row['Range_AD']=='inside' and (best_name!='MLR' or row['Leverage_AD']=='inside') else 'outside/warning'
    ad.append(row)
# MLR equation on standardized and raw descriptors
mlr.fit(Xtr_all[:,mlr_idxs],ytr)
sc=mlr.named_steps['scaler']; lr=mlr.named_steps['lr']
coef_std=lr.coef_; int_std=lr.intercept_; coef_raw=coef_std/sc.scale_; int_raw=int_std-np.sum(coef_std*sc.mean_/sc.scale_)
mlr_eq={'features':mlr_features,'intercept_standardized':float(int_std),'coef_standardized':{f:float(c) for f,c in zip(mlr_features,coef_std)},'intercept_raw':float(int_raw),'coef_raw':{f:float(c) for f,c in zip(mlr_features,coef_raw)},'vif':{f:float(v) for f,v in zip(mlr_features, vif(sc.transform(Xtr_all[:,mlr_idxs])))} }
# Plots
for m,info in models.items():
    est=info['est'].fit(info['Xtr'],ytr); ptr=est.predict(info['Xtr']); pte=est.predict(info['Xte'])
    for setname,y,p in [('TS',ytr,ptr),('VS',yte,pte)]:
        plt.figure(figsize=(6,5)); plt.scatter(y,p); lo=min(y.min(),p.min())-0.2; hi=max(y.max(),p.max())+0.2; plt.plot([lo,hi],[lo,hi],linestyle='--')
        plt.xlabel('LogKd obserwowany'); plt.ylabel('LogKd przewidywany'); plt.title(f'{m}: observed vs predicted ({setname})'); plt.tight_layout(); plt.savefig(f'{OUTDIR}/obs_pred_{m.replace(" ","_")}_{setname}.png',dpi=220); plt.close()
    yall=np.r_[ytr,yte]; pall=np.r_[ptr,pte]
    plt.figure(figsize=(6,5)); plt.scatter(pall,yall-pall); plt.axhline(0,linestyle='--'); plt.xlabel('LogKd przewidywany'); plt.ylabel('Reszta (obs - pred)'); plt.title(f'{m}: wykres reszt'); plt.tight_layout(); plt.savefig(f'{OUTDIR}/residuals_{m.replace(" ","_")}.png',dpi=220); plt.close()
plt.figure(figsize=(7,4.5)); labels=[r['Model'] for r in summary]; x=np.arange(len(labels)); w=0.35
plt.bar(x-w/2,[r['RMSECV'] for r in summary],w,label='RMSECV'); plt.bar(x+w/2,[r['RMSEP'] for r in summary],w,label='RMSEP'); plt.xticks(x,labels); plt.ylabel('Błąd LogKd'); plt.title('Porównanie błędów modeli'); plt.legend(); plt.tight_layout(); plt.savefig(f'{OUTDIR}/model_comparison_errors.png',dpi=220); plt.close()
if best_name=='MLR':
    std_res=(ytr-ptr_best)/resid_sd if resid_sd else ytr*0
    std_te=(yte-pte_best)/resid_sd if resid_sd else yte*0
    plt.figure(figsize=(6,5)); plt.scatter(lev_tr,std_res,label='TS'); plt.scatter(lev_te,std_te,marker='s',label='VS');
    if lev_ext is not None: plt.scatter(lev_ext,[0]*len(lev_ext),marker='D',label='True external')
    plt.axvline(h_star,linestyle='--'); plt.axhline(3,linestyle='--'); plt.axhline(-3,linestyle='--'); plt.xlabel('Leverage h'); plt.ylabel('Standaryzowana reszta'); plt.title('Williams plot - MLR'); plt.legend(); plt.tight_layout(); plt.savefig(f'{OUTDIR}/applicability_domain_williams.png',dpi=220); plt.close()
else:
    dte=min_dist(Xtr_std,Xte_std); dext=min_dist(Xtr_std,Xext_std) if not np.isnan(Xext_std).any() else []
    plt.figure(figsize=(6,5)); plt.scatter(dte,yte-pte_best,marker='s',label='VS');
    if len(dext): plt.scatter(dext,[0]*len(dext),marker='D',label='True external')
    plt.axvline(dist_thr,linestyle='--'); plt.axhline(0,linestyle='--'); plt.xlabel('Odległość do najbliższego TS'); plt.ylabel('Reszta / EXT bez obserwacji'); plt.title(f'Domena stosowalności - {best_name}'); plt.legend(); plt.tight_layout(); plt.savefig(f'{OUTDIR}/applicability_domain_distance.png',dpi=220); plt.close()
# Save tables
for fname,data in [('model_summary.csv',summary),('predictions.csv',predictions),('true_external.csv',true_ext),('applicability_domain.csv',ad),('mlr_top_subsets.csv',[{k:v for k,v in s.items() if k!='idxs'} for s in subsets_sorted[:10]])]:
    if data:
        keys=list(data[0].keys())
        with open(f'{OUTDIR}/{fname}','w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(data)
# Combined results json
results={'diagnostics':{'n_records_total_excluding_header':len(records),'n_modelling_compounds':len(model),'n_TS':int(tr.sum()),'n_VS':int(te.sum()),'n_EXT_rows_in_excel':len(ext),'response_col':y_col,'descriptor_cols_total':len(desc),'numeric_descriptor_cols':len(desc)-len(set(nonnum)),'non_numeric_descriptor_cols':len(set(nonnum)),'descriptor_cols_with_missing_modelling':len(missing_cols),'missing_cols_sample':missing_cols[:20],'constant_cols_count':len(const),'constant_cols_sample':const[:20],'near_constant_cols_count':len(near_const),'near_constant_cols_sample':near_const[:20],'duplicate_compound_names':dup_names,'response_outliers_IQR':resp_out,'candidate_feature_z_gt_3':zout},'candidate_features':features,'candidate_corr_TS':corr.tolist(),'mlr_subset_top10':[{k:v for k,v in s.items() if k!='idxs'} for s in subsets_sorted[:10]],'model_summary':summary,'best_model_name':best_name,'best_features':best_info['features'],'best_params':best_info['params'],'predictions':predictions,'true_external':true_ext,'ad':{'distance_threshold':dist_thr,'h_star':h_star,'rows':ad},'mlr_equation':mlr_eq}
with open(f'{OUTDIR}/results.json','w',encoding='utf-8') as f: json.dump(results,f,ensure_ascii=False,indent=2)
print(json.dumps({'best':best_name,'best_features':best_info['features'],'summary':summary,'true_external':true_ext,'diagnostics':results['diagnostics'],'mlr_equation':mlr_eq},ensure_ascii=False,indent=2))
