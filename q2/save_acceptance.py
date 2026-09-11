"""第二问验收入口；输出统一保存到result/verification。"""
from pathlib import Path
import hashlib,json,unittest
from q2_config import HERE

def main():
    target=HERE/'result/verification'
    target.mkdir(parents=True,exist_ok=True)
    suite=unittest.TestSuite()
    for name in ['test_q2','test_q2_integration','test_q2_fast','test_q2_seed','test_q2_ml','test_q2_bounded_seed']:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
    with (target/'acceptance_tests.txt').open('w',encoding='utf-8') as f:
        result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    report=dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                skipped=len(result.skipped),passed=result.wasSuccessful())
    (target/'acceptance_tests.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    protected=target/'protected_originals.json'
    if protected.exists():
        for filename,expected in json.loads(protected.read_text(encoding='utf-8')).items():
            assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==expected,filename
    print(json.dumps(report))
    if not result.wasSuccessful():
        raise SystemExit(1)

if __name__=='__main__':
    main()
