import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from selrisk.realdata import ROOT,ensure_dataset,feature_names,load_regional,validate_regional,verify_file
from selrisk.research import design,fit_model,lookup_date,score_rows,train_research
from selrisk.models.fusion import classify_risk,daily_risk,trigger_level
from selrisk.dataset import build_terrain_grid

@pytest.mark.parametrize("kind,n,p",[("power",14610,137),("stations",20088,102)])
def test_contract(kind,n,p):
    d,f,_=load_regional(kind)
    assert len(d)==n and d.target.eq(1).sum()==p and not d.target.eq(0).any()
    out=design(d)
    pd.testing.assert_series_equal(out.target,d.target)
    assert not (out.research_role.eq("unlabeled_background")&d.date_uncertainty_overlap).any()
    positive=pd.to_datetime(d.loc[d.target.eq(1),"date"])
    bg=pd.to_datetime(out.loc[out.research_role.eq("unlabeled_background"),"date"])
    assert all(not ((bg-t).abs()<=pd.Timedelta(days=7)).any() for t in positive)

@pytest.mark.parametrize("mutation",["zero","status","duplicate","gap","split","negative","infinite","season","complete","cutoff","rain","temperature","feature_leak","date","bool","missing"])
def test_invalid_contract(mutation):
    d,f,_=load_regional("power");d=d.copy()
    i=100
    if mutation=="zero":d.loc[i,"target"]=0
    elif mutation=="status":d.loc[i,"label_status"]="negative"
    elif mutation=="duplicate":d.loc[i,"date"]=d.loc[i-1,"date"]
    elif mutation=="gap":d=d.drop(index=i)
    elif mutation=="split":d.loc[i,"split"]="test"
    elif mutation=="negative":d.loc[i,f[0]]=-1
    elif mutation=="infinite":d.loc[i,f[0]]=np.inf
    elif mutation=="season":d.loc[i,"day_of_year_sin"]=.123
    elif mutation=="complete":d.loc[i,"feature_complete"]=False
    elif mutation=="cutoff":d.loc[i,"feature_window_end_utc"]="2099-01-01"
    elif mutation=="rain":d.loc[i,"precipitation_prev3d_mm"]=1e7
    elif mutation=="temperature":d.loc[i,"temperature_min_lag2utc_c"]=100
    elif mutation=="feature_leak":f=f+["target"]
    elif mutation=="date":d.loc[i,"date"]=None
    elif mutation=="bool":
        d["date_uncertainty_overlap"]=d.date_uncertainty_overlap.astype(object)
        d.loc[i,"date_uncertainty_overlap"]=None
    elif mutation=="missing":d=d.drop(columns=[f[0]])
    with pytest.raises((ValueError,KeyError,TypeError)):
        validate_regional(d,f)

def test_no_implicit_demo():
    with pytest.raises(RuntimeError,match="disabled"):
        build_terrain_grid()

def test_missing_not_safe():
    grid=pd.DataFrame({"basin":["A","B"],"susceptibility":[.5,np.nan]})
    weather=pd.DataFrame({"date":["2020-01-01"],"basin":["A"],"trigger_prob":[.3]})
    out=daily_risk(grid,weather,"2030-01-01")
    assert out.risk.isna().all()
    assert out.risk_class.eq("нет данных").all()
    for fn in (classify_risk,trigger_level):
        assert all(x=="нет данных" for x in fn([np.nan,np.inf,-np.inf]))
    weather=pd.concat([weather,weather])
    with pytest.raises(ValueError,match="Duplicate"):
        daily_risk(grid,weather,"2020-01-01")

def test_lookup_unknown_date_and_domain():
    d,_,_=load_regional()
    m=json.loads((ROOT/"web/data/model_power.json").read_text())
    result=lookup_date(d,m,"2030-01-01")
    assert result["score"] is None and result["probability"] is None
    assert result["status"]=="no_data"
    x=d.iloc[[100]].copy();x.loc[:,m["features"][0]]=1e9
    scores,status=score_rows(m,x)
    assert np.isnan(scores[0]) and status[0]=="out_of_training_domain"

def test_fit_never_uses_test_or_validation():
    d,f,_=load_regional();a=design(d)
    first=fit_model(a,f,.3)
    changed=a.copy()
    changed.loc[~changed.split.eq("train"),f]=123456.
    second=fit_model(changed,f,.3)
    np.testing.assert_array_equal(first[0].mean_,second[0].mean_)
    np.testing.assert_array_equal(first[-1].coef_,second[-1].coef_)
    assert first[0].n_samples_seen_==int((a.split.eq("train")&a.evaluation_eligible).sum())

def test_bad_archive_rejected(tmp_path):
    p=tmp_path/"bad.zip";p.write_bytes(b"broken")
    with pytest.raises(ValueError,match="modified"):
        ensure_dataset(tmp_path/"extract",p)


@pytest.mark.parametrize("member",["/absolute.txt","selrisk-dataset-v0.2/../../escape","wrong-prefix/file"])
def test_archive_traversal_rejected(tmp_path,monkeypatch,member):
    import selrisk.realdata as rd
    manifest=json.loads((ROOT/"datasets/manifest.json").read_text())
    p=tmp_path/"unsafe.zip"
    with zipfile.ZipFile(p,"w") as z:
        z.writestr(member,"do not extract")
    monkeypatch.setattr(rd,"sha256",lambda path:manifest["sha256"])
    with pytest.raises(ValueError,match="Unsafe"):
        ensure_dataset(tmp_path/"out",p)


def test_modified_checksum_manifest_rejected(tmp_path):
    (tmp_path/"checksums.sha256").write_text("0"*64+"  data/forged.csv\n")
    with pytest.raises(ValueError,match="checksum manifest"):
        verify_file(tmp_path,"data/forged.csv")

def test_checksums_all_dataset_files():
    root=ensure_dataset()
    for line in (root/"checksums.sha256").read_text().splitlines():
        digest,rel=line.split("  ",1)
        verify_file(root,rel)

def test_data_rerun_reproducible(tmp_path):
    report=train_research("power",tmp_path)
    expected=json.loads((ROOT/"web/data/report_power.json").read_text())
    assert report==expected
    assert (tmp_path/"model_power.json").read_bytes()==(ROOT/"web/data/model_power.json").read_bytes()
