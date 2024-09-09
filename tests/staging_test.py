import requests
import json

test_sample = json.dumps({
    'data': [
        [2013, 8, 3000.0, 3200.0], 
        [2013, 9, 3200.0, 3000.0],  
        [2013, 10, 3100.0, 3200.0]  
    ]
    })
test_sample = str(test_sample)

def test_ml_service(scoreurl, scorekey):
    assert scoreurl != None

    if scorekey is None:
        headers = {'Content-Type':'application/json'}
    else:
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer ' + scorekey)}

    resp = requests.post(scoreurl, test_sample, headers=headers)
    assert resp.status_code == requests.codes.ok
    assert resp.text != None
    assert resp.headers.get('content-type') == 'application/json'
    assert int(resp.headers.get('Content-Length')) > 0
