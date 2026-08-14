from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import csv, json, math, os, shutil, subprocess
import numpy as np
from scipy import sparse

TWSTREAM_REPO='https://github.com/Du-Team/TWStream.git'
TWSTREAM_COMMIT='4e084d1ce29f116fc9896ffb270640d8fb24348f'

@dataclass(frozen=True)
class TWStreamOfficialResult:
    labels: np.ndarray
    coverage: float
    point_to_microcluster: np.ndarray
    microcluster_to_cluster: dict[int,int]
    cluster_probability: dict[int,float]
    repository_commit: str
    output_dir: str
    parameters: dict
    def metadata(self):
        d=asdict(self);d.pop('labels');d.pop('point_to_microcluster');return d

class TWStreamOfficialAdapter:
    """Adapter for the authors' Java TWStream implementation.

    This adapter intentionally remains separate from the project's structural
    common-evaluator comparator.  It runs the authors' native three-way engine
    and preserves unassigned/outlier points as label -1 rather than inventing a
    100%-coverage mapping.
    """
    def __init__(self, repo_dir: Path, work_dir: Path):
        self.repo_dir=Path(repo_dir);self.work_dir=Path(work_dir)

    @staticmethod
    def toolchain_status() -> dict[str,bool]:
        return {name:shutil.which(name) is not None for name in ['git','mvn','java','javac']}

    def setup(self) -> Path:
        missing=[k for k,v in self.toolchain_status().items() if not v]
        if missing: raise RuntimeError(f'Missing TWStream official toolchain programs: {missing}')
        if not (self.repo_dir/'.git').exists():
            self.repo_dir.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['git','clone',TWSTREAM_REPO,str(self.repo_dir)],check=True)
        subprocess.run(['git','fetch','--all','--tags'],cwd=self.repo_dir,check=True)
        subprocess.run(['git','checkout','--detach',TWSTREAM_COMMIT],cwd=self.repo_dir,check=True)
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.repo_dir,text=True).strip()
        if head!=TWSTREAM_COMMIT: raise RuntimeError(f'TWStream checkout mismatch: {head}')
        subprocess.run(['mvn','-q','-DskipTests','package'],cwd=self.repo_dir,check=True)
        jars=sorted((self.repo_dir/'target').glob('TWStream-*.jar'),key=lambda p:p.stat().st_size,reverse=True)
        if not jars: raise FileNotFoundError('TWStream Maven build produced no target/TWStream-*.jar')
        return jars[0]

    def _bridge_source(self) -> str:
        return r'''import edu.jsnu.sunjr.TWStream;
public class TWStreamBridge {
  public static void main(String[] args) {
    if (args.length != 12) throw new IllegalArgumentException("expected 12 args");
    String input=args[0], output=args[1];
    int dim=Integer.parseInt(args[2]); double lambda=Double.parseDouble(args[3]);
    double radius=Double.parseDouble(args[4]); double beta=Double.parseDouble(args[5]);
    int gap=Integer.parseInt(args[6]); int k=Integer.parseInt(args[7]);
    int alpha=Integer.parseInt(args[8]); double tau=Double.parseDouble(args[9]);
    int n=Integer.parseInt(args[10]); int outputInterval=Integer.parseInt(args[11]);
    TWStream model=new TWStream(dim,lambda,radius,beta,gap,k,alpha,tau,n,outputInterval);
    model.process(input,output,n,outputInterval);
  }
}
'''

    def _compile_bridge(self, jar: Path) -> Path:
        bridge=self.work_dir/'bridge';bridge.mkdir(parents=True,exist_ok=True)
        src=bridge/'TWStreamBridge.java';src.write_text(self._bridge_source(),encoding='utf-8')
        subprocess.run(['javac','-cp',str(jar),str(src)],check=True)
        cls=bridge/'TWStreamBridge.class'
        if not cls.exists(): raise RuntimeError('TWStream bridge compilation failed')
        return bridge

    @staticmethod
    def official_beta(lambda_: float) -> float:
        # Same initialization shown in the authors' TWStream.java main method.
        return 1.0-math.pow(2.0,-float(lambda_))+0.0001

    @staticmethod
    def _write_input_csv(path: Path, x, block_rows: int = 512) -> tuple[int,int]:
        """Write dense or CSR input without materializing a full sparse stream."""
        if sparse.issparse(x):
            x=x.tocsr(); n,d=map(int,x.shape)
            if n==0 or d==0: raise ValueError('x must be a nonempty 2-D matrix')
            with path.open('w',encoding='utf-8',newline='') as f:
                for start in range(0,n,int(block_rows)):
                    block=x[start:min(start+int(block_rows),n)].toarray()
                    np.savetxt(f,block,delimiter=',',fmt='%.17g')
            return n,d
        a=np.asarray(x,dtype=np.float64)
        if a.ndim!=2 or len(a)==0: raise ValueError('x must be a nonempty 2-D matrix')
        np.savetxt(path,a,delimiter=',',fmt='%.17g')
        return int(a.shape[0]),int(a.shape[1])

    def run(self, x, *, radius: float, lambda_: float, k: int, tau: float,
            beta: float|None=None, gap_time: int=100, alpha: int=2) -> TWStreamOfficialResult:
        jar=self.setup();bridge=self._compile_bridge(jar)
        run_dir=self.work_dir/'run';
        if run_dir.exists(): shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True,exist_ok=True)
        input_csv=run_dir/'input.csv'
        n_rows,n_dim=self._write_input_csv(input_csv,x)
        beta=self.official_beta(lambda_) if beta is None else float(beta)
        cp=os.pathsep.join([str(bridge),str(jar)])
        # output interval > n suppresses intermediate MC2Cluster files; final is always written.
        subprocess.run(['java','-cp',cp,'TWStreamBridge',str(input_csv),str(run_dir),str(n_dim),str(lambda_),str(radius),str(beta),str(gap_time),str(k),str(alpha),str(tau),str(n_rows),str(n_rows+1)],check=True)
        p2m=run_dir/'point2MC.csv';m2c=run_dir/'MC2Clusterfinal.csv'
        if not p2m.exists() or not m2c.exists(): raise FileNotFoundError('Official TWStream did not produce expected final mapping files')
        point_mc=np.full(n_rows,-1,np.int64)
        with p2m.open(newline='') as f:
            for row in csv.reader(f):
                if len(row)>=2: point_mc[int(row[0])]=int(row[1])
        mc_cluster={};mc_prob={}
        with m2c.open(newline='') as f:
            for row in csv.reader(f):
                if len(row)>=2:
                    mc=int(row[0]);mc_cluster[mc]=int(row[1]);mc_prob[mc]=float(row[2]) if len(row)>2 else float('nan')
        labels=np.asarray([mc_cluster.get(int(mc),-1) for mc in point_mc],np.int64)
        coverage=float(np.mean(labels>=0))
        params={'radius':float(radius),'lambda':float(lambda_),'beta':float(beta),'gap_time':int(gap_time),'k':int(k),'alpha':int(alpha),'tau':float(tau)}
        (run_dir/'adapter_manifest.json').write_text(json.dumps({'repository':TWSTREAM_REPO,'commit':TWSTREAM_COMMIT,'coverage':coverage,'parameters':params},indent=2),encoding='utf-8')
        return TWStreamOfficialResult(labels,coverage,point_mc,mc_cluster,mc_prob,TWSTREAM_COMMIT,str(run_dir),params)
