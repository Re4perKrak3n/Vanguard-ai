/**
 * Sentinel-Brain Service Worker
 * Enables PWA install, offline support, and push notifications.
 */

const CACHE_NAME = 'sentinel-brain-v1';
const ASSETS = [
    '/',
    '/static/style.css',
    '/static/app.js',
    '/manifest.json',
];

// ── Install ────────────────────────────────────────────────────────
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.addAll(ASSETS);
        })
    );
    self.skipWaiting();
});

// ── Activate ───────────────────────────────────────────────────────
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((names) => {
            return Promise.all(
                names
                    .filter((name) => name !== CACHE_NAME)
                    .map((name) => caches.delete(name))
            );
        })
    );
    self.clients.claim();
});

// ── Fetch (Network-first, fallback to cache) ───────────────────────
self.addEventListener('fetch', (event) => {
    // Skip WebSocket and API requests
    if (
        event.request.url.includes('/ws') ||
        event.request.url.includes('/api/')
    ) {
        return;
    }

    event.respondWith(
        fetch(event.request)
            .then((response) => {
                // Cache successful responses
                if (response.status === 200) {
                    const clone = response.clone();
                    caches.open(CACHE_NAME).then((cache) => {
                        cache.put(event.request, clone);
                    });
                }
                return response;
            })
            .catch(() => {
                // Fallback to cache
                return caches.match(event.request);
            })
    );
});

// ── Push Notifications ─────────────────────────────────────────────
self.addEventListener('push', (event) => {
    const data = event.data ? event.data.json() : {};

    const options = {
        body: data.message || 'New security alert',
        icon: '/static/icon-192.png',
        badge: '/static/icon-192.png',
        vibrate: [200, 100, 200, 100, 200],
        tag: 'sentinel-alert',
        renotify: true,
        actions: [
            { action: 'view', title: 'View' },
            { action: 'dismiss', title: 'Dismiss' },
        ],
    };

    event.waitUntil(
        self.registration.showNotification('🚨 Sentinel Alert', options)
    );
});

self.addEventListener('notificationclick', (event) => {
    event.notification.close();

    if (event.action === 'dismiss') return;

    event.waitUntil(
        clients.matchAll({ type: 'window' }).then((clientList) => {
            // Focus existing window or open new one
            for (const client of clientList) {
                if (client.url.includes(self.location.origin) && 'focus' in client) {
                    return client.focus();
                }
            }
            return clients.openWindow('/');
        })
    );
});
