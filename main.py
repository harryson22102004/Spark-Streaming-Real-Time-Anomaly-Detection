import numpy as np
from collections import deque
import time
 
# PySpark streaming simulation (without cluster)
class MicroBatch:
  def __init__(self, data, timestamp): self.data=data; self.ts=timestamp
 
class StreamingWindow:
    def __init__(self, duration=60, slide=10):
        self.duration=duration; self.slide=slide
        self.batches=deque()
    def add(self, batch):
        self.batches.append(batch)
        cutoff=batch.ts-self.duration
        while self.batches and self.batches[0].ts<cutoff: self.batches.popleft()
    def get_window_data(self):
        return [x for b in self.batches for x in b.data]
 
class IsolationForestSimple:
    """Simplified isolation forest for streaming anomaly detection."""
    def __init__(self, n_trees=10, max_depth=8):
        self.n_trees=n_trees; self.max_depth=max_depth; self.trees=[]
    def fit(self, X):
        self.trees=[]
        for _ in range(self.n_trees):
            idx=np.random.choice(len(X), min(256,len(X)), replace=False)
            self.trees.append(self._build(X[idx], 0))
    def _build(self, X, depth):
        if depth>=self.max_depth or len(X)<=1: return {'size':len(X)}
        col=np.random.randint(X.shape[1])
        lo,hi=X[:,col].min(),X[:,col].max()
        if lo>=hi: return {'size':len(X)}
        split=np.random.uniform(lo,hi)
        return {'col':col,'split':split,'depth':depth,
                'left':self._build(X[X[:,col]<split],depth+1),
                'right':self._build(X[X[:,col]>=split],depth+1)}
    def path_length(self, x, node, current=0):
        if 'size' in node: return current+np.log2(max(node['size'],1))
        return self.path_length(x, node['left'] if x[node['col']]<node['split'] else node['right'], current+1)
    def anomaly_score(self, x):
        avg_path=np.mean([self.path_length(x,t) for t in self.trees])
        n=256; H=np.log(n-1)+0.5772; c=2*H-2*(n-1)/n
        return 2**(-avg_path/c)
 
window=StreamingWindow(60,10)
iforest=IsolationForestSimple()
normal=np.random.randn(200,3)
iforest.fit(normal)
total_anomalies=0
for batch_id in range(20):
    ts=batch_id*10
    data=np.random.randn(50,3)
    if batch_id in [5,12]: data[10:15]*=5  # inject anomalies
    batch=MicroBatch(data,ts); window.add(batch)
    scores=[iforest.anomaly_score(x) for x in data]
    anomalies=sum(s>0.65 for s in scores)
    total_anomalies+=anomalies
    if anomalies>0: print(f"Batch {batch_id:2d} (t={ts:3d}s): {anomalies} anomalies detected (max_score={max(scores):.2f})")
print(f"Total anomalies: {total_anomalies}")
