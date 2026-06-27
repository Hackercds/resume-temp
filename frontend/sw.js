// PWA Service Worker - 离线缓存静态资源
const CACHE_NAME = 'resume-rag-v1';
const STATIC_ASSETS = [
    './',
    './index.html',
    './src/styles/main.css',
    './src/main.js',
    './src/api/client.js',
    './manifest.json'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.addAll(STATIC_ASSETS);
        }).catch(() => {})
    );
    self.skipWaiting();
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) => {
            return Promise.all(
                keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))
            );
        })
    );
    self.clients.claim();
});

self.addEventListener('fetch', (event) => {
    // API 请求不缓存
    if (event.request.url.includes('/api/') || event.request.url.includes('/health')) {
        return;
    }
    event.respondWith(
        caches.match(event.request).then((cached) => {
            return cached || fetch(event.request).then((response) => {
                return caches.open(CACHE_NAME).then((cache) => {
                    cache.put(event.request, response.clone());
                    return response;
                });
            });
        }).catch(() => caches.match('./index.html'))
    );
});
