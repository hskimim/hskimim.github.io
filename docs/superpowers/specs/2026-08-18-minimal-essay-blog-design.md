# hskimim.github.io — Minimal Essay Blog Design

**Date:** 2026-08-18
**Status:** Approved in conversation; this document records the validated design.

## Purpose

A personal technical blog in the spirit of lilianweng.github.io: long-form
English essays with math, code, and figures. The design philosophy is
minimalist — typography does all the work, everything else gets out of the way.

## Goals

- Writing workflow: add one markdown file, commit, push — site deploys itself.
- Long-form essay support: KaTeX math, syntax-highlighted code, images with
  captions, table of contents for long posts.
- Near-zero JavaScript; pure static HTML output that loads instantly.
- Minimal visual design: one typeface, near-monochrome palette, narrow
  reading column.

## Non-Goals

- No comments, tags cloud, search, analytics, or newsletter.
- No dark-mode toggle (system `prefers-color-scheme` only).
- No migration of the old 2019–2020 Jekyll blog (deliberately deleted).

## Stack

- **Astro** (static output). Chosen over Hugo (Go templating makes custom
  design iteration slower) and Jekyll (dated toolchain). Node required
  locally for `npm run dev` / `npm run build`.
- Markdown pipeline: `remark-math` + `rehype-katex` for math, built-in Shiki
  for code highlighting.

## Repository & Deployment

- GitHub repo `hskimim/hskimim.github.io`, local clone at
  `/Users/mark/projects/hskimim.github.io` (independent from the `mark`
  monorepo).
- GitHub Actions workflow using the official `withastro/action`: push to
  `main` → build → deploy to GitHub Pages at `https://hskimim.github.io`.

## Site Structure

| Route | Content |
|-------|---------|
| `/` | One–two line intro + post list (date and title only) |
| `/posts/<slug>/` | Essay page: title, date, TOC for long posts, body |
| `/about/` | Short bio page |
| `/rss.xml` | RSS feed |
| `/sitemap-index.xml` | Sitemap |
| `404` | Minimal not-found page |

## Content Model

- Posts live in `src/content/posts/*.md`.
- Frontmatter: `title`, `date`, `description`, optional `draft`.
- A post is published by pushing a markdown file to `main`.

## Design

- Narrow reading column (~65ch), generous line height.
- Near-monochrome palette; a single accent color for links.
- One high-quality body typeface with system-font fallback; 2–3 candidates
  will be rendered during implementation for the author to pick from.
- Dark mode follows `prefers-color-scheme` automatically; no toggle.
- Header: name + about link only. No logo, sidebar, or footer clutter.

## Initial Content & Verification

- Seed with one typography-specimen post exercising math, code blocks,
  blockquotes, and captioned images — serves as design check and writing
  template; deletable once real posts exist.
- Verification: `npm run build` passes locally, deploy succeeds, and
  `https://hskimim.github.io` renders correctly in the browser.
