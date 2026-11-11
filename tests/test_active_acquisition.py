import json
import unittest
from bluebridge.active_acquisition import prepare_plans, prepare_answers
from test_workflows import fixture
from bluebridge.workflows import initial_state


class ActiveAcquisition(unittest.TestCase):
    def test_plan_has_only_eligible_public_input(self):
        e=fixture();e['reference']='SECRET';p=prepare_plans(e,initial_state(e),0)
        text=json.dumps(p['jobs']);self.assertNotIn('SECRET',text);self.assertNotIn('s2c',text)
        self.assertEqual(len(p['jobs']),2)

    def test_invalid_plan_is_explicitly_charged_fallback(self):
        e=fixture();s=initial_state(e);p=prepare_plans(e,s,0)
        a=prepare_answers(e,s,0,p,[])
        self.assertTrue(all(x['planning_error'] for x in a['sidecars']))
        self.assertEqual(len(a['jobs']),2)

    def test_bounded_queries_and_unknown_output(self):
        e=fixture();s=initial_state(e);p=prepare_plans(e,s,0)
        with self.assertRaises(ValueError):prepare_answers(e,s,0,p,[{'job_id':'unexpected','text':'{}'}])
        o={'job_id':p['jobs'][0]['job_id'],'text':json.dumps({'queries':['revenue']*4})}
        a=prepare_answers(e,s,0,p,[o]);self.assertTrue(a['sidecars'][0]['planning_error'])

    def test_plan_is_not_injected_as_factual_evidence(self):
        e=fixture();s=initial_state(e);p=prepare_plans(e,s,0)
        o={'job_id':p['jobs'][0]['job_id'],'text':json.dumps({'findings':['INVENTED ASSERTION'],'queries':['FY2024 revenue']})}
        a=prepare_answers(e,s,0,p,[o]);self.assertNotIn('INVENTED ASSERTION',json.dumps(a['jobs']))
        self.assertIn('unverified_plan',a['sidecars'][0])


if __name__=='__main__':unittest.main()
