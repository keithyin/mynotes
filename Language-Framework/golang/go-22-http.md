# http.Client

# CookieJar
https://developer.mozilla.org/en-US/docs/Web/HTTP/Cookies
https://web.dev/same-site-same-origin/

这个是http.Client用来记录访问网站的cookie的地方。
在请求的时候，带上对应的cookie，在返回的时候，更新对应的cookie。

```go
type CookieJar interface {
	// SetCookies handles the receipt of the cookies in a reply for the
	// given URL.  It may or may not choose to save the cookies, depending
	// on the jar's policy and implementation.
	SetCookies(u *url.URL, cookies []*Cookie)

	// Cookies returns the cookies to send in a request for the given URL.
	// It is up to the implementation to honor the standard cookie use
	// restrictions such as in RFC 6265.
	Cookies(u *url.URL) []*Cookie
}
```

CookieJar其实是一个interface，具体行为可以由使用者自己定义。

* SetCookies：返回的 cookie 如果设置
* Cookies：访问一个地址时，应该带着哪些 cookie

## SetCookies

拿 `http/cookiejar/jar.go` 中 `MemoryCookies`举例

cookies主要存储在 `entries map[string]map[string]entry` 中，这里重要的就是两个key表是什么含义
* first key: eTLD+1  [什么是etld](https://web.dev/same-site-same-origin/)。实际上就是 site
* subkey: cookie的 `name/domain/path`


```go
func (j *Jar) setCookies(u *url.URL, cookies []*http.Cookie, now time.Time) {
	if len(cookies) == 0 {
		return
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return
	}
	host, err := canonicalHost(u.Host)
	if err != nil {
		return
	}
	key := jarKey(host, j.psList)
	defPath := defaultPath(u.Path)

	j.mu.Lock()
	defer j.mu.Unlock()

	submap := j.entries[key] // map[string]map[string]entry

	modified := false
  // 遍历传入的 cookie，如果曾经存在：删除+添加。不存在：添加
	for _, cookie := range cookies { 
		e, remove, err := j.newEntry(cookie, now, defPath, host)
		if err != nil {
			continue
		}
		id := e.id()
		if remove {
			if submap != nil {
				if _, ok := submap[id]; ok {
					delete(submap, id)
					modified = true
				}
			}
			continue
		}
		if submap == nil {
			submap = make(map[string]entry)
		}

		if old, ok := submap[id]; ok {
			e.Creation = old.Creation
			e.seqNum = old.seqNum
		} else {
			e.Creation = now
			e.seqNum = j.nextSeqNum
			j.nextSeqNum++
		}
		e.LastAccess = now
		submap[id] = e
		modified = true
	}

	if modified {
		if len(submap) == 0 {
			delete(j.entries, key)
		} else {
			j.entries[key] = submap
		}
	}
}
```

# 参考资料
https://developer.mozilla.org/en-US/docs/Web/HTTP/Cookies
https://web.dev/same-site-same-origin/
