import copy
import json
import unittest
from bluebridge.workflows import (initial_state, prepare_event, reduce_event, source_ids, digest,
                       impact_route, apply_output, selective_jobs, packet)


def fixture():
    return {'episode_id':'example', 'issuer':'TEST', 'initial_as_of':'2025-01-01',
        'sources':[{'source_id':'s0','published_at':'2025-01-01','url':'https://example.com/0'},
                   {'source_id':'s1','published_at':'2025-04-01','url':'https://example.com/1'},
                   {'source_id':'s2','published_at':'2025-07-01','url':'https://example.com/2'}],
        'chunks':[{'chunk_id':'s0c','source_id':'s0','text':'TEST FY2024 revenue was USD 100 million.','locator':'p1'},
                  {'chunk_id':'s1c','source_id':'s1','text':'TEST Q1 revenue was USD 110 million.','locator':'p1'},
                  {'chunk_id':'s2c','source_id':'s2','text':'TEST Q2 revenue was USD 120 million.','locator':'p1'}],
        'initial_claims':[{'claim_id':'hist','instruction':'Keep this fixed historical FY2024 revenue fact.',
                           'text':'FY2024 revenue was USD 100 million.','citations':['s0c'], 'metric_spec':{}},
                          {'claim_id':'rolling','instruction':'Update the latest quarter revenue.',
                           'text':'Latest revenue was USD 100 million.','citations':['s0c'],'metric_spec':{}}],
        'events':[{'event_id':'e1','as_of':'2025-04-01','new_source_ids':['s1']},
                  {'event_id':'e2','as_of':'2025-07-01','new_source_ids':['s2']}]}


def output(job, value=110, patch=None, citation='s1c', quote='TEST Q1 revenue was USD 110 million.'):
    return {'job_id':job, 'text':json.dumps({'tool':'finish','disposition':'update',
        'value':value,'unit':'USD million','citations':[citation],
        'reason':'A new quarter arrived.','patch':patch or f'Latest revenue was USD {value} million.',
        'evidence_quotes':[{'chunk_id':citation,'quote':quote}], 'calculations':[]})}


class Workflows(unittest.TestCase):
    def test_discovery_and_cutoff_no_future(self):
        e=fixture();self.assertEqual(source_ids(e,0),['s0','s1'])
        e['reference']={'answer':'SECRET_GOLD'}
        e['chunks'][0]['expected_answer']='SECRET_LABEL'
        b=prepare_event(e,initial_state(e),0,'fixed_zero')
        rendered=json.dumps(b['jobs'])
        for forbidden in ('SECRET_GOLD','SECRET_LABEL','USD 120','s2c'):
            self.assertNotIn(forbidden,rendered)

    def test_cache_router_is_conservative(self):
        e=fixture();s=initial_state(e)
        self.assertFalse(impact_route(e['initial_claims'][0],s['hist'],e,0)['inspect'])
        self.assertTrue(impact_route(e['initial_claims'][1],s['rolling'],e,0)['inspect'])
        e['chunks'][1]['text'] += ' Revised prior figures.'
        self.assertTrue(impact_route(e['initial_claims'][0],s['hist'],e,0)['inspect'])

    def test_invalid_response_carries_actual_error_not_gold(self):
        e=fixture();s=initial_state(e);s['rolling']['text']='WRONG PREVIOUS VALUE 999'
        b=prepare_event(e,s,0,'incremental_tools');r=reduce_event(e,s,0,b,[])
        self.assertEqual(r['state']['rolling']['text'],'WRONG PREVIOUS VALUE 999')
        self.assertTrue(r['state']['rolling']['unresolved'])
        next_jobs=prepare_event(e,r['state'],1,'incremental_tools')['jobs']
        self.assertIn('WRONG PREVIOUS VALUE 999',json.dumps(next_jobs))

    def test_model_wrong_value_still_propagates_and_is_flagged(self):
        e=fixture();s=initial_state(e);b=prepare_event(e,s,0,'incremental_tools')
        r=reduce_event(e,s,0,b,[output(b['jobs'][0]['job_id'],999)])
        self.assertIn('999',r['state']['rolling']['text'])
        self.assertTrue(r['state']['rolling']['unresolved'])
        self.assertIn('proposal_value_not_quoted_or_calculated',r['records'][0]['audit']['flags'])

    def test_state_tamper_and_duplicate_outputs_rejected(self):
        e=fixture();s=initial_state(e);b=prepare_event(e,s,0,'incremental_tools');o=output(b['jobs'][0]['job_id'])
        with self.assertRaises(ValueError):reduce_event(e,s,0,b,[o,o])
        s['rolling']['text']='tampered'
        with self.assertRaises(ValueError):reduce_event(e,s,0,b,[])

    def test_job_sidecar_schema_and_no_input_mutation(self):
        e=fixture();s=initial_state(e);old=copy.deepcopy([e,s])
        b=prepare_event(e,s,0,'fixed_tools')
        self.assertEqual([e,s],old)
        self.assertTrue(all(set(j)=={'job_id','messages','max_new_tokens'} for j in b['jobs']))

    def test_unread_and_future_citations_cannot_pass(self):
        e=fixture();s=initial_state(e);claim=e['initial_claims'][1]
        r,meta=apply_output(e,claim,s['rolling'],0,output('x',120,citation='s2c',quote=e['chunks'][2]['text']),read_ids=['s1c'])
        self.assertTrue(r['unresolved'])
        self.assertIn('unknown_or_ineligible_citation:s2c',meta['audit']['flags'])
        r,meta=apply_output(e,claim,s['rolling'],0,output('x',100,citation='s0c',quote=e['chunks'][0]['text']),read_ids=['s1c'])
        self.assertIn('unknown_or_ineligible_citation:s0c',meta['audit']['flags'])

    def test_selective_recheck_is_flag_based_not_label_based(self):
        e=fixture();s=initial_state(e);b=prepare_event(e,s,0,'incremental_selective')
        r=reduce_event(e,s,0,b,[output(b['jobs'][0]['job_id'],999)])
        checks=selective_jobs(e,s,0,b,r)
        self.assertEqual(len(checks['jobs']),1)
        r=reduce_event(e,s,0,b,[output(b['jobs'][0]['job_id'])])
        self.assertEqual(selective_jobs(e,s,0,b,r)['jobs'],[])

    def test_full_packet_has_all_eligible_chunks_and_no_truncation(self):
        e=fixture();s=initial_state(e)
        chunks,cost=packet(e,e['initial_claims'][0],s['hist'],0,full=True)
        self.assertEqual({c['chunk_id'] for c in chunks},{'s0c','s1c'})
        self.assertEqual(cost['selected_chars'],sum(len(c['text'])for c in chunks))


if __name__=='__main__':unittest.main()
