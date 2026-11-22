"""An alias training may only use the bound idle-budget class of machine at its live rate."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from alias_queue import pricing,close_owned
from gpu import GPUError


class Hub:
    id='bound'
    config={'power_conf_id':'active-spec'}
    def __init__(self,price=1.68,cash=45.38,holds=0,spec='active-spec'):
        self.price,self.cash,self.holds,self.spec=price,cash,holds,spec
    def machine(self):
        return {'state':'Stopped','powerConfId':'active-spec','powerConfDto':{'id':self.spec,'cardNum':1,'displayName':'RTX 4090','price':self.price},'ssdPower':0,'diskPower':0}
    def call(self,op):
        assert op=='balance'
        return {'balances':[{'productId':'heros','cash':{'postedBalance':self.cash,'reservedAmount':self.holds}}]}


def test_queue_uses_live_price_for_entire_window_and_subtracts_reserved_cash():
    _,v=pricing(Hub(price=1.5,cash=8,holds=3));assert v['available_cash_cny']==5 and v['estimate_ceiling_cny']==4.5 and v['max_minutes']==180


@pytest.mark.parametrize('hub',[Hub(price=1.69),Hub(cash=10,holds=5),Hub(spec='different')])
def test_refuse_price_cap_insufficient_available_cash_or_wrong_spec(hub):
    with pytest.raises(GPUError):pricing(hub)


def test_remeasured_full_observation_window_still_respects_actual_cash():
    assert pricing(Hub(cash=6.72),240)[1]['estimate_ceiling_cny']==6.72
    with pytest.raises(GPUError):pricing(Hub(cash=8,holds=1.3),240)
    with pytest.raises(GPUError):pricing(Hub(),300)


def test_transient_preclose_ssh_read_rechecks_ownership(monkeypatch):
    import alias_queue as q
    m={'state':'Running','jobId':'ours'};h=Hub();count=[];events=[];monkeypatch.setattr(h,'machine',lambda:dict(m))
    def close(hub,current,timeout):
        count.append(current['jobId'])
        if len(count)==1:raise GPUError('SSH/GPU probe failed (exit 255); GPU usage is unknown')
        return dict(current,state='Stopped')
    monkeypatch.setattr(q.gpu,'close',close);monkeypatch.setattr(q.time,'sleep',lambda _:None)
    assert close_owned(h,m,lambda **kw:events.append(kw))['state']=='Stopped' and count==['ours','ours'] and len(events)==1


def test_ambiguous_close_or_different_owner_is_not_retried(monkeypatch):
    import alias_queue as q
    h=Hub();m={'state':'Running','jobId':'ours'};monkeypatch.setattr(h,'machine',lambda:dict(m));count=[]
    def err(*args):count.append(1);raise GPUError('close result unknown')
    monkeypatch.setattr(q.gpu,'close',err)
    with pytest.raises(GPUError,match='unknown'):close_owned(h,m,lambda **kw:None)
    monkeypatch.setattr(h,'machine',lambda:dict(m,jobId='someone_else'))
    with pytest.raises(GPUError,match='changed'):close_owned(h,m,lambda **kw:None)
    assert count==[1]
