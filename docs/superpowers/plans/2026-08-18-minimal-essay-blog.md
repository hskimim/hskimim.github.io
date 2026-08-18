# Minimal Essay Blog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy a minimalist, typography-first English essay blog (Astro static site) to https://hskimim.github.io via GitHub Pages.

**Architecture:** Astro 5 static site with a single content collection (`posts`) of markdown files. One base layout + one global stylesheet carry the entire design. Markdown pipeline renders KaTeX math and Shiki-highlighted code at build time, so the shipped site has zero client-side JavaScript. GitHub Actions builds and deploys on every push to `main`.

**Tech Stack:** Astro ^5, @astrojs/rss, @astrojs/sitemap, remark-math + rehype-katex + katex, Shiki (built into Astro), @fontsource-variable fonts, GitHub Pages + Actions (`withastro/action`).

**Spec:** `docs/superpowers/specs/2026-08-18-minimal-essay-blog-design.md`

## Global Constraints

- Site URL is exactly `https://hskimim.github.io`; base path is `/`.
- Content language is English; every page sets `lang="en"`.
- Zero client-side JavaScript in the built output (no `<script>` tags in `dist/`).
- No comments, tag cloud, search, analytics, newsletter, or dark-mode toggle.
- Dark mode via `prefers-color-scheme` only.
- All colors and fonts come from CSS custom properties defined once in `src/styles/global.css`; never hardcode a color elsewhere.
- Reading column: `max-width: 42rem` on `body`; nothing renders wider (code blocks and display math scroll inside `overflow-x: auto`).
- Node >= 20 required for local builds.
- Project root: `/Users/mark/projects/hskimim.github.io` (its own git repo, branch `main`, already initialized with the spec committed).

## File Structure

```
/Users/mark/projects/hskimim.github.io/
├── .github/workflows/deploy.yml     # Task 6 — build & deploy to Pages
├── .gitignore                       # Task 1
├── astro.config.mjs                 # Task 1 — site, sitemap, markdown pipeline
├── package.json                     # Task 1
├── tsconfig.json                    # Task 1
├── public/
│   ├── robots.txt                   # Task 5
│   └── images/specimen-figure.svg   # Task 3
└── src/
    ├── consts.ts                    # Task 1 — SITE_TITLE, SITE_DESCRIPTION
    ├── utils.ts                     # Task 4 — formatDate()
    ├── content.config.ts            # Task 3 — posts collection schema
    ├── content/posts/
    │   └── typography-specimen.md   # Task 3 — seed post
    ├── layouts/BaseLayout.astro     # Task 2 — html shell, header nav
    ├── styles/global.css            # Task 2 — the entire design
    └── pages/
        ├── index.astro              # Task 1 placeholder → Task 4 post list
        ├── about.astro              # Task 4
        ├── 404.astro                # Task 4
        ├── rss.xml.js               # Task 5
        └── posts/[id].astro         # Task 3 — essay page
```

---

### Task 1: Project scaffold and build pipeline

**Files:**
- Create: `package.json`, `astro.config.mjs`, `tsconfig.json`, `.gitignore`, `src/consts.ts`, `src/pages/index.astro`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `SITE_TITLE: string` and `SITE_DESCRIPTION: string` exported from `src/consts.ts`; a working `npm run build` that outputs to `dist/`; markdown pipeline config (remark-math, rehype-katex, Shiki dual themes) that Task 3 relies on.

- [ ] **Step 1: Verify Node version**

Run: `node --version`
Expected: `v20.x` or higher. If lower, stop and report — do not work around it.

- [ ] **Step 2: Write package.json**

```json
{
  "name": "hskimim.github.io",
  "type": "module",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "astro dev",
    "build": "astro build",
    "preview": "astro preview"
  },
  "dependencies": {
    "@astrojs/rss": "^4.0.0",
    "@astrojs/sitemap": "^3.2.0",
    "@fontsource-variable/inter": "^5.2.0",
    "@fontsource-variable/newsreader": "^5.2.0",
    "@fontsource-variable/source-serif-4": "^5.2.0",
    "astro": "^5.0.0",
    "katex": "^0.16.0",
    "rehype-katex": "^7.0.0",
    "remark-math": "^6.0.0"
  }
}
```

- [ ] **Step 3: Write astro.config.mjs**

```js
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: 'https://hskimim.github.io',
  integrations: [sitemap()],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark' },
    },
  },
});
```

- [ ] **Step 4: Write tsconfig.json**

```json
{
  "extends": "astro/tsconfigs/strict",
  "include": [".astro/types.d.ts", "**/*"],
  "exclude": ["dist"]
}
```

- [ ] **Step 5: Write .gitignore**

```
node_modules/
dist/
.astro/
.DS_Store
```

- [ ] **Step 6: Write src/consts.ts**

```ts
export const SITE_TITLE = 'hskimim';
export const SITE_DESCRIPTION = 'Long-form technical essays.';
```

- [ ] **Step 7: Write placeholder src/pages/index.astro**

(Task 2 replaces this with the layout-based version; this exists only so the build has a page.)

```astro
---
import { SITE_TITLE } from '../consts';
---
<html lang="en">
  <head><meta charset="utf-8" /><title>{SITE_TITLE}</title></head>
  <body><h1>{SITE_TITLE}</h1></body>
</html>
```

- [ ] **Step 8: Install and build**

Run: `cd /Users/mark/projects/hskimim.github.io && npm install && npm run build`
Expected: install succeeds; build ends with a line like `Complete!` and `dist/index.html` exists (`test -f dist/index.html && echo OK` prints `OK`).

- [ ] **Step 9: Commit**

```bash
git add package.json package-lock.json astro.config.mjs tsconfig.json .gitignore src/
git commit -m "feat: scaffold Astro project with math/code markdown pipeline"
```

---

### Task 2: Global typography stylesheet and base layout

**Files:**
- Create: `src/styles/global.css`, `src/layouts/BaseLayout.astro`
- Modify: `src/pages/index.astro` (use the layout)

**Interfaces:**
- Consumes: `SITE_TITLE`, `SITE_DESCRIPTION` from `src/consts.ts` (Task 1).
- Produces: `BaseLayout.astro` with props `{ title: string; description?: string }` and a default `<slot />` — every page in Tasks 3–5 wraps content in it. CSS classes later tasks use: `.post-list`, `.toc`, `.intro`.

- [ ] **Step 1: Write src/styles/global.css**

```css
:root {
  --bg: #ffffff;
  --fg: #1a1a1a;
  --muted: #6b6b6b;
  --accent: #1d4ed8;
  --rule: #e5e5e5;
  --code-bg: #f6f6f6;
  --font-body: 'Newsreader Variable', Georgia, 'Times New Roman', serif;
  --font-mono: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #121212;
    --fg: #e6e6e6;
    --muted: #9a9a9a;
    --accent: #8ab4f8;
    --rule: #2a2a2a;
    --code-bg: #1e1e1e;
  }
  .astro-code,
  .astro-code span {
    color: var(--shiki-dark) !important;
    background-color: var(--shiki-dark-bg) !important;
  }
}

* {
  box-sizing: border-box;
}

html {
  -webkit-text-size-adjust: 100%;
}

body {
  margin: 0 auto;
  max-width: 42rem;
  padding: 2rem 1.25rem 4rem;
  background: var(--bg);
  color: var(--fg);
  font-family: var(--font-body);
  font-size: 1.125rem;
  line-height: 1.7;
}

header nav {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 3.5rem;
}

header nav a {
  color: var(--fg);
  text-decoration: none;
}

a {
  color: var(--accent);
  text-decoration-thickness: 1px;
  text-underline-offset: 2px;
}

h1 {
  font-size: 1.9rem;
  line-height: 1.25;
  margin: 0 0 0.5rem;
}

h2 {
  font-size: 1.4rem;
  line-height: 1.3;
  margin: 2.5rem 0 0.75rem;
}

h3 {
  font-size: 1.15rem;
  margin: 2rem 0 0.5rem;
}

time {
  color: var(--muted);
  font-size: 0.9rem;
  font-variant-numeric: tabular-nums;
}

.intro {
  color: var(--muted);
  margin-bottom: 2.5rem;
}

.post-list {
  list-style: none;
  margin: 0;
  padding: 0;
}

.post-list li {
  display: flex;
  gap: 1.25rem;
  align-items: baseline;
  margin-bottom: 0.9rem;
}

.post-list time {
  white-space: nowrap;
}

.toc {
  border-left: 2px solid var(--rule);
  padding-left: 1rem;
  margin: 2rem 0;
  font-size: 0.95rem;
}

.toc ul {
  list-style: none;
  margin: 0;
  padding: 0;
}

blockquote {
  border-left: 2px solid var(--rule);
  margin: 1.5rem 0;
  padding-left: 1rem;
  color: var(--muted);
}

figure {
  margin: 2rem 0;
  text-align: center;
}

figcaption {
  color: var(--muted);
  font-size: 0.9rem;
  margin-top: 0.5rem;
}

img {
  max-width: 100%;
  height: auto;
}

pre.astro-code {
  padding: 1rem;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 0.85rem;
  line-height: 1.6;
}

code {
  font-family: var(--font-mono);
}

:not(pre) > code {
  font-size: 0.85em;
  background: var(--code-bg);
  padding: 0.1em 0.35em;
  border-radius: 4px;
}

.katex-display {
  overflow-x: auto;
  overflow-y: hidden;
  padding: 0.25rem 0;
}

hr {
  border: none;
  border-top: 1px solid var(--rule);
  margin: 2.5rem 0;
}
```

- [ ] **Step 2: Write src/layouts/BaseLayout.astro**

```astro
---
import '@fontsource-variable/newsreader';
import 'katex/dist/katex.min.css';
import '../styles/global.css';
import { SITE_TITLE, SITE_DESCRIPTION } from '../consts';

interface Props {
  title: string;
  description?: string;
}

const { title, description = SITE_DESCRIPTION } = Astro.props;
---

<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="description" content={description} />
    <link
      rel="alternate"
      type="application/rss+xml"
      title={SITE_TITLE}
      href={new URL('rss.xml', Astro.site)}
    />
    <title>{title}</title>
  </head>
  <body>
    <header>
      <nav>
        <a href="/">{SITE_TITLE}</a>
        <a href="/about/">about</a>
      </nav>
    </header>
    <main>
      <slot />
    </main>
  </body>
</html>
```

- [ ] **Step 3: Rewrite src/pages/index.astro to use the layout**

```astro
---
import BaseLayout from '../layouts/BaseLayout.astro';
import { SITE_TITLE } from '../consts';
---

<BaseLayout title={SITE_TITLE}>
  <p class="intro">Post list coming in Task 4.</p>
</BaseLayout>
```

- [ ] **Step 4: Build and verify**

Run: `npm run build && grep -c 'href="/about/"' dist/index.html && grep -c '<script' dist/index.html || true`
Expected: build succeeds; first grep prints `1` (header nav present); second grep prints `0` (no client JS). Note: `grep -c` exits non-zero on zero matches — the trailing `|| true` keeps the shell happy; what matters is the printed counts.

- [ ] **Step 5: Commit**

```bash
git add src/styles/global.css src/layouts/BaseLayout.astro src/pages/index.astro
git commit -m "feat: add global typography styles and base layout"
```

---

### Task 3: Content collection, specimen post, and essay page

**Files:**
- Create: `src/content.config.ts`, `src/content/posts/typography-specimen.md`, `public/images/specimen-figure.svg`, `src/pages/posts/[id].astro`

**Interfaces:**
- Consumes: `BaseLayout.astro` props `{ title, description }` (Task 2); markdown pipeline from `astro.config.mjs` (Task 1).
- Produces: content collection `posts` with schema `{ title: string; date: Date; description: string; draft: boolean }`; post URLs of shape `/posts/<id>/` where `<id>` is the markdown filename without extension. Tasks 4–5 query it via `getCollection('posts', ({ data }) => !data.draft)`.

- [ ] **Step 1: Write src/content.config.ts**

```ts
import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const posts = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/posts' }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    description: z.string(),
    draft: z.boolean().default(false),
  }),
});

export const collections = { posts };
```

- [ ] **Step 2: Write public/images/specimen-figure.svg**

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 200" width="400" height="200">
  <rect width="400" height="200" fill="none"/>
  <polyline points="20,180 90,140 160,150 230,80 300,95 380,30"
    fill="none" stroke="#888" stroke-width="2"/>
  <line x1="20" y1="180" x2="380" y2="180" stroke="#bbb" stroke-width="1"/>
  <line x1="20" y1="180" x2="20" y2="20" stroke="#bbb" stroke-width="1"/>
</svg>
```

- [ ] **Step 3: Write src/content/posts/typography-specimen.md**

````markdown
---
title: "Typography Specimen"
date: 2026-08-18
description: "A sample essay exercising every element this blog can render. Delete once real posts exist."
---

This post exists to exercise every element an essay here can contain.
It doubles as a writing template — copy its frontmatter, delete it when
real posts exist.

## Prose and emphasis

Body text sits in a narrow column at a comfortable line height. Inline
elements include *emphasis*, **strong emphasis**, `inline code`, and
[links](https://example.com). Quotations render like this:

> The purpose of computing is insight, not numbers.
> — Richard Hamming

## Mathematics

Inline math flows with the text: the loss $\mathcal{L}(\theta) = -\sum_i y_i \log \hat{y}_i$
should sit on the baseline. Display math gets its own block:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

## Code

```python
def attention(q, k, v):
    scores = q @ k.T / math.sqrt(k.shape[-1])
    return softmax(scores) @ v
```

## Figures

<figure>
  <img src="/images/specimen-figure.svg" alt="A sample line chart" />
  <figcaption>Fig. 1. Figures carry numbered captions in muted text.</figcaption>
</figure>

## Lists

1. Ordered lists for sequences.
2. Second item.

- Unordered lists for everything else.
- Second item.
````

- [ ] **Step 4: Write src/pages/posts/[id].astro**

```astro
---
import { getCollection, render } from 'astro:content';
import BaseLayout from '../../layouts/BaseLayout.astro';
import { formatDate } from '../../utils';

export async function getStaticPaths() {
  const posts = await getCollection('posts', ({ data }) => !data.draft);
  return posts.map((post) => ({ params: { id: post.id }, props: { post } }));
}

const { post } = Astro.props;
const { Content, headings } = await render(post);
const toc = headings.filter((h) => h.depth === 2);
---

<BaseLayout title={post.data.title} description={post.data.description}>
  <article>
    <h1>{post.data.title}</h1>
    <time datetime={post.data.date.toISOString()}>{formatDate(post.data.date)}</time>
    {
      toc.length >= 3 && (
        <nav class="toc">
          <ul>
            {toc.map((h) => (
              <li>
                <a href={`#${h.slug}`}>{h.text}</a>
              </li>
            ))}
          </ul>
        </nav>
      )
    }
    <Content />
  </article>
</BaseLayout>
```

- [ ] **Step 5: Write src/utils.ts** (needed by Step 4's import; Task 4 reuses it)

```ts
export function formatDate(date: Date): string {
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}
```

- [ ] **Step 6: Build and verify math, code, TOC render**

Run:
```bash
npm run build
grep -c 'class="katex"' dist/posts/typography-specimen/index.html
grep -c 'astro-code' dist/posts/typography-specimen/index.html
grep -c 'class="toc"' dist/posts/typography-specimen/index.html
grep -c '<script' dist/posts/typography-specimen/index.html || true
```
Expected: build succeeds; katex count >= 2, astro-code count >= 1, toc count = 1, script count = 0.

- [ ] **Step 7: Commit**

```bash
git add src/content.config.ts src/content/ src/pages/posts/ src/utils.ts public/images/
git commit -m "feat: add posts collection, essay page, and typography specimen"
```

---

### Task 4: Home post list, about page, 404

**Files:**
- Modify: `src/pages/index.astro`
- Create: `src/pages/about.astro`, `src/pages/404.astro`

**Interfaces:**
- Consumes: `posts` collection and `/posts/<id>/` URL shape (Task 3); `formatDate` from `src/utils.ts` (Task 3); `BaseLayout` (Task 2).
- Produces: final public pages; no downstream code depends on their internals.

- [ ] **Step 1: Rewrite src/pages/index.astro with the post list**

```astro
---
import { getCollection } from 'astro:content';
import BaseLayout from '../layouts/BaseLayout.astro';
import { SITE_TITLE } from '../consts';
import { formatDate } from '../utils';

const posts = (await getCollection('posts', ({ data }) => !data.draft)).sort(
  (a, b) => b.data.date.valueOf() - a.data.date.valueOf()
);
---

<BaseLayout title={SITE_TITLE}>
  <p class="intro">Long-form essays on machine learning and markets.</p>
  <ul class="post-list">
    {
      posts.map((post) => (
        <li>
          <time datetime={post.data.date.toISOString()}>{formatDate(post.data.date)}</time>
          <a href={`/posts/${post.id}/`}>{post.data.title}</a>
        </li>
      ))
    }
  </ul>
</BaseLayout>
```

- [ ] **Step 2: Write src/pages/about.astro**

(Placeholder copy — the author will edit the text themselves later. Do not invent biography details.)

```astro
---
import BaseLayout from '../layouts/BaseLayout.astro';
---

<BaseLayout title="About">
  <h1>About</h1>
  <p>
    I'm hskimim. I write long-form essays here. Find me on
    <a href="https://github.com/hskimim">GitHub</a>.
  </p>
</BaseLayout>
```

- [ ] **Step 3: Write src/pages/404.astro**

```astro
---
import BaseLayout from '../layouts/BaseLayout.astro';
---

<BaseLayout title="Not found">
  <h1>404</h1>
  <p>This page doesn't exist. <a href="/">Back to the essays.</a></p>
</BaseLayout>
```

- [ ] **Step 4: Build and verify**

Run:
```bash
npm run build
grep -c 'href="/posts/typography-specimen/"' dist/index.html
test -f dist/about/index.html && test -f dist/404.html && echo PAGES_OK
```
Expected: build succeeds; grep prints `1`; `PAGES_OK` printed.

- [ ] **Step 5: Commit**

```bash
git add src/pages/index.astro src/pages/about.astro src/pages/404.astro
git commit -m "feat: add home post list, about, and 404 pages"
```

---

### Task 5: RSS feed, sitemap, robots.txt

**Files:**
- Create: `src/pages/rss.xml.js`, `public/robots.txt`

**Interfaces:**
- Consumes: `posts` collection (Task 3); `SITE_TITLE`, `SITE_DESCRIPTION` (Task 1); sitemap integration already configured in `astro.config.mjs` (Task 1).
- Produces: `/rss.xml` and `/sitemap-index.xml` in the built site.

- [ ] **Step 1: Write src/pages/rss.xml.js**

```js
import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import { SITE_TITLE, SITE_DESCRIPTION } from '../consts';

export async function GET(context) {
  const posts = (await getCollection('posts', ({ data }) => !data.draft)).sort(
    (a, b) => b.data.date.valueOf() - a.data.date.valueOf()
  );
  return rss({
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
    site: context.site,
    items: posts.map((post) => ({
      title: post.data.title,
      description: post.data.description,
      pubDate: post.data.date,
      link: `/posts/${post.id}/`,
    })),
  });
}
```

- [ ] **Step 2: Write public/robots.txt**

```
User-agent: *
Allow: /

Sitemap: https://hskimim.github.io/sitemap-index.xml
```

- [ ] **Step 3: Build and verify**

Run:
```bash
npm run build
grep -c '<item>' dist/rss.xml
test -f dist/sitemap-index.xml && test -f dist/robots.txt && echo FEEDS_OK
```
Expected: build succeeds; `<item>` count >= 1; `FEEDS_OK` printed.

- [ ] **Step 4: Commit**

```bash
git add src/pages/rss.xml.js public/robots.txt
git commit -m "feat: add RSS feed and robots.txt"
```

---

### Task 6: GitHub repository, Actions deploy, live verification

**Files:**
- Create: `.github/workflows/deploy.yml`

**Interfaces:**
- Consumes: the complete buildable site (Tasks 1–5); `gh` CLI authenticated as `hskimim` (already logged in); local repo on branch `main`.
- Produces: live site at `https://hskimim.github.io`.

- [ ] **Step 1: Write .github/workflows/deploy.yml**

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Build with Astro
        uses: withastro/action@v3

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: Commit the workflow**

```bash
git add .github/workflows/deploy.yml
git commit -m "ci: deploy to GitHub Pages on push to main"
```

- [ ] **Step 3: Create the GitHub repository and push**

Run:
```bash
cd /Users/mark/projects/hskimim.github.io
gh repo create hskimim/hskimim.github.io --public --description "Long-form technical essays" --source . --push
```
Expected: repo created, `main` pushed. If `--source`/`--push` flags fail, fall back to:
```bash
gh repo create hskimim/hskimim.github.io --public --description "Long-form technical essays"
git remote add origin https://github.com/hskimim/hskimim.github.io.git
git push -u origin main
```

- [ ] **Step 4: Set Pages source to GitHub Actions**

Run: `gh api -X POST repos/hskimim/hskimim.github.io/pages -f build_type=workflow`
Expected: JSON response with `"build_type": "workflow"`. If it returns 409 (already exists), run `gh api -X PUT repos/hskimim/hskimim.github.io/pages -f build_type=workflow` instead.

- [ ] **Step 5: Watch the deploy run**

Run: `gh run watch --repo hskimim/hskimim.github.io --exit-status $(gh run list --repo hskimim/hskimim.github.io --limit 1 --json databaseId --jq '.[0].databaseId')`
Expected: run completes with success. If the first push raced the Pages enablement and the deploy job failed, re-run it: `gh workflow run deploy.yml --repo hskimim/hskimim.github.io` and watch again.

- [ ] **Step 6: Verify the live site**

Run:
```bash
curl -sL https://hskimim.github.io | grep -c 'href="/posts/typography-specimen/"'
curl -s -o /dev/null -w '%{http_code}\n' https://hskimim.github.io/posts/typography-specimen/
```
Expected: `1`, then `200`. GitHub's CDN can take a minute or two after the first deploy — retry a couple of times before treating a 404 as a failure.

---

### Task 7: Font selection checkpoint (user decision — run in main session, not a subagent)

**Files:**
- Modify: `src/styles/global.css` (the `--font-body` line), `src/layouts/BaseLayout.astro` (the fontsource import line)

**Interfaces:**
- Consumes: live site + local dev server; the three fontsource packages installed in Task 1.
- Produces: final `--font-body` choice; the two unused fontsource packages removed from `package.json`.

This task needs the author's eyes, so the main session drives it: render the specimen post under each candidate, show the user, apply their pick.

- [ ] **Step 1: Produce three variants of the specimen post**

For each candidate, set the import in `BaseLayout.astro` and the token in `global.css`, run `npm run build`, and screenshot `dist/posts/typography-specimen/index.html` (serve via `npm run preview`, capture with browser tools):

| Candidate | Import | `--font-body` |
|-----------|--------|----------------|
| Newsreader (serif, current default) | `@fontsource-variable/newsreader` | `'Newsreader Variable', Georgia, 'Times New Roman', serif` |
| Source Serif 4 (serif, sturdier) | `@fontsource-variable/source-serif-4` | `'Source Serif 4 Variable', Georgia, 'Times New Roman', serif` |
| Inter (sans, Lilian-adjacent) | `@fontsource-variable/inter` | `'Inter Variable', -apple-system, 'Helvetica Neue', Arial, sans-serif` |

- [ ] **Step 2: Show the user all three and ask for a pick**

Present the three screenshots side by side (AskUserQuestion with the candidates).

- [ ] **Step 3: Apply the pick and clean up**

Set the chosen import + token permanently. Remove the two losing `@fontsource-variable/*` packages: `npm uninstall <loser-1> <loser-2>`. Run `npm run build` — expected: success.

- [ ] **Step 4: Commit and push**

```bash
git add package.json package-lock.json src/styles/global.css src/layouts/BaseLayout.astro
git commit -m "design: finalize body typeface"
git push
```

- [ ] **Step 5: Confirm live**

Run: `curl -s https://hskimim.github.io/posts/typography-specimen/ | grep -ci '<chosen font family name>'`
Expected: >= 1 after the deploy finishes.
