#!/usr/bin/env node

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const sourcePath = join(root, "data/manifests/unsplash_exterior_mvp.sources.json");
const destination = join(root, "data/source/unsplash_exteriors");
const manifestPath = join(root, "data/manifests/unsplash_exterior_mvp.json");
const sources = JSON.parse(readFileSync(sourcePath, "utf8"));

mkdirSync(destination, { recursive: true });

const runCurl = (args) => execFileSync("curl", args, { encoding: "utf8" });
const decodeHtml = (value) => value.replaceAll("&amp;", "&");
const now = new Date().toISOString();
const assets = [];

for (const [index, source] of sources.entries()) {
  const html = runCurl(["--location", "--fail", "--silent", "--show-error", source.source_page_url]);
  const match = html.match(/<meta property="og:image" content="([^"]+)"/);
  if (!match) throw new Error(`Missing og:image metadata for ${source.source_id}`);

  const directImageBaseUrl = decodeHtml(match[1]).split("?")[0];
  const datasetNumber = index + 1;
  const filename = `${String(datasetNumber).padStart(3, "0")}.jpg`;
  const outputPath = join(destination, filename);
  const legacyPath = join(destination, `${source.source_id}.jpg`);
  const downloadUrl = `${directImageBaseUrl}?auto=format&fit=max&w=2048&q=90`;

  if (!existsSync(outputPath) && existsSync(legacyPath)) {
    renameSync(legacyPath, outputPath);
  }

  if (!existsSync(outputPath)) {
    execFileSync("curl", ["--location", "--fail", "--silent", "--show-error", "--max-time", "90", "--output", outputPath, downloadUrl]);
  }
  const bytes = readFileSync(outputPath);

  assets.push({
    asset_id: `unsplash_${source.source_id}`,
    dataset_number: datasetNumber,
    source_id: source.source_id,
    source_page_url: source.source_page_url,
    direct_image_url: directImageBaseUrl,
    local_path: `data/source/unsplash_exteriors/${basename(outputPath)}`,
    sha256: createHash("sha256").update(bytes).digest("hex"),
    retrieved_at: now,
    rights_basis: "Unsplash License",
    license_url: "https://unsplash.com/license",
    derivative_use_allowed: true,
    portfolio_display_allowed: true,
    listing_group_id: null,
    split: "unassigned",
    dataset_role: source.dataset_role ?? "primary_candidate",
    exclusion_reason: source.exclusion_reason ?? null,
    notes: "Individual stock exterior image. Do not treat this asset as part of a multi-photo listing. Review depicted trademarks, artworks, and privacy/property-rights considerations before public use."
  });
}

writeFileSync(manifestPath, `${JSON.stringify({ generated_at: now, assets }, null, 2)}\n`);
console.log(`Downloaded ${assets.length} images to ${destination}`);
console.log(`Wrote manifest to ${manifestPath}`);
