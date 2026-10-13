// Run with: node --test tests/dashboard-security.cjs (requires playwright).
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

test('labels remain inert through editing, storage, modal, export and re-export', async () => {
    const browser = await chromium.launch({ channel: 'msedge', headless: true });
    try {
        const source = fs.readFileSync(process.env.DASHBOARD_PATH || path.join(__dirname, '../fund-launch-dashboard.html'), 'utf8');
        const errors = [];
        async function open(html) {
            const context = await browser.newContext({ acceptDownloads: true });
            await context.route('**/*', route => {
                if (route.request().url() === 'http://dashboard.test/') {
                    return route.fulfill({ contentType: 'text/html', body: html });
                }
                if (route.request().url() === 'https://cdn.tailwindcss.com/') {
                    // Keep tests offline; supply only the utility needed for modal interaction.
                    return route.fulfill({ contentType: 'application/javascript', body:
                        "const style = document.createElement('style'); style.textContent = '.hidden { display: none !important; }'; document.head.appendChild(style);" });
                }
                return route.abort();
            });
            const page = await context.newPage();
            page.on('pageerror', error => errors.push(error.message));
            page.on('dialog', dialog => dialog.dismiss());
            await page.goto('http://dashboard.test/');
            return page;
        }
        async function download(page) {
            const pending = page.waitForEvent('download');
            await page.locator('#export-data-btn').click();
            return fs.readFileSync(await (await pending).path(), 'utf8');
        }
        const payload = `" autofocus onfocus="window.labelExecuted=1"><img data-label-attack src=x onerror="window.labelExecuted=1"><script data-label-attack>window.labelExecuted=1</script></ScRiPt><script data-label-attack>window.labelExecuted=1</script><!-- & ' $& $\u0060 $'    `;
        const page = await open(source);
        const taskId = await page.locator('.task-name-input-timeline').first().evaluate(el => el.closest('tr').dataset.taskId);
        const task = page.locator(`tr[data-task-id="${taskId}"] .task-name-input-timeline`);
        const expense = page.locator('.item-name-input').first();
        await task.fill(payload);
        await task.dispatchEvent('change');
        await expense.fill(payload);
        await expense.dispatchEvent('change');
        await page.reload();
        async function assertInert(p) {
            assert.equal(await p.locator('[data-label-attack]').count(), 0);
            assert.equal(await p.evaluate(() => window.labelExecuted), undefined);
            assert.equal(await p.locator(`tr[data-task-id="${taskId}"] .task-name-input-timeline`).inputValue(), payload);
            assert.equal(await p.locator('.item-name-input').first().inputValue(), payload);
            await p.locator('.manage-schedule-btn').first().click();
            assert.equal(await p.locator('#modal-content-host h2').textContent(), 'Payment Schedule for: ' + payload);
            assert.equal(await p.locator('[data-label-attack]').count(), 0);
            await p.locator('#save-schedule-btn').click();
        }
        await assertInert(page);
        await page.locator('#aum-input').focus();
        await page.locator('#aum-input').fill('12345678');
        await page.locator('#aum-input').dispatchEvent('change');
        // Transient live DOM must never be included in a shared export.
        await page.evaluate(() => {
            const node = document.createElement('div');
            node.id = 'transient-node';
            node.setAttribute('onclick', 'window.labelExecuted=1');
            document.body.appendChild(node);
        });
        const exported = await download(page);
        assert.ok(!exported.includes('id="transient-node"'));
        assert.ok(exported.includes('\\u003c'));
        const recipient = await open(exported);
        await assertInert(recipient);
        assert.equal(await recipient.locator('#aum-input').inputValue(), '12,345,678');
        const secondRecipient = await open(await download(recipient));
        await assertInert(secondRecipient);
        // Empty plans must round-trip without resurrecting the defaults.
        await secondRecipient.evaluate(() => { timelineTasks = []; costItems = []; renderAll(); });
        const empty = await open(await download(secondRecipient));
        assert.equal(await empty.locator('.task-name-input-timeline, .item-name-input').count(), 0);
        assert.deepEqual(errors, []);
    } finally {
        await browser.close();
    }
});
