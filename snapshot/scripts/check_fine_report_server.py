"""Check video seek support and deny paths outside the explicit report list."""
import urllib.request
import urllib.error


def main():
    base='http://127.0.0.1:8011'
    request=urllib.request.Request(base+'/video/s4_adl_15',headers={'Range':'bytes=0-63'})
    with urllib.request.urlopen(request) as r:
        assert r.status==206 and len(r.read())==64
        assert r.headers['Content-Range'].startswith('bytes 0-63/')
        assert r.headers['Content-Type']=='video/mp4'
    with urllib.request.urlopen(urllib.request.Request(base+'/video/s4_adl_15',method='HEAD')) as r:
        assert r.status==200 and int(r.headers['Content-Length'])>0 and not r.read()
    for path in ['/config/live_actions.json','/../config/live_actions.json','/video/s1_adl_01','/video/../../config/live_actions.json']:
        try:urllib.request.urlopen(base+path)
        except urllib.error.HTTPError as error:assert error.code==404
        else:raise AssertionError('Non-allowlisted path served: '+path)
    print('PASS: video range, HEAD, and four path isolation checks')


if __name__=='__main__':main()
