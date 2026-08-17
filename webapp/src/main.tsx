import { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type Tab = "studio" | "edit" | "quality" | "training";
type Photo = { file: File; url: string; area: string; viewpoint: string };
type Bridge = { from_index: number; to_index: number; kind: "spatial" | "lighting"; duration_seconds: number };
type Qa = { source_path: string; decision: string; reason_codes: string[] };
type Runtime = { environment: { machine: string; capabilities: { mps_available: boolean; cuda_available: boolean } }; profiles: Record<string, { device: string; compatible: boolean; warning?: string }> };

const api = (path: string) => `${import.meta.env.DEV ? "/api" : ""}${path}`;

function App() {
  const [tab, setTab] = useState<Tab>("studio");
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [bridges, setBridges] = useState<Bridge[]>([]);
  const [bridgeFrom, setBridgeFrom] = useState(0);
  const [bridgeTo, setBridgeTo] = useState(1);
  const [bridgeKind, setBridgeKind] = useState<Bridge["kind"]>("spatial");
  const [bridgeDuration, setBridgeDuration] = useState(3);
  const [duration, setDuration] = useState(12);
  const [dissolve, setDissolve] = useState(0.5);
  const [profile, setProfile] = useState("local_mps");
  const [runtime, setRuntime] = useState<Runtime>();
  const [status, setStatus] = useState("Add a property set to begin.");
  const [qa, setQa] = useState<Qa[]>([]);
  const [video, setVideo] = useState<string>();
  const [editPrompt, setEditPrompt] = useState("Convert this daylight exterior to a natural premium warm dusk scene; preserve exact building geometry, windows, landscaping, driveway, and vertical lines.");
  const [editing, setEditing] = useState(false);
  const [editSource, setEditSource] = useState(0);
  const [loraDataset, setLoraDataset] = useState<File>();
  const [loraEvidence, setLoraEvidence] = useState<File>();
  const [loraResult, setLoraResult] = useState<{ training_permitted: boolean; reason_codes: string[]; pair_counts: Record<string, number> }>();

  useEffect(() => { fetch(api("/runtime")).then((r) => r.json()).then(setRuntime).catch(() => setStatus("Start the local API to inspect GPU compatibility.")); }, []);
  useEffect(() => () => photos.forEach((photo) => URL.revokeObjectURL(photo.url)), [photos]);

  const selectedRuntime = runtime?.profiles[profile];
  const bridgePairs = useMemo(() => new Set(bridges.map((b) => `${b.from_index}-${b.to_index}`)), [bridges]);

  function addPhotos(files: FileList | null) {
    if (!files) return;
    const incoming = [...files].slice(0, 12 - photos.length).map((file) => ({ file, url: URL.createObjectURL(file), area: "unclassified", viewpoint: "unclassified" }));
    setPhotos((current) => [...current, ...incoming]);
    setStatus("Label each view, then select only compatible pairs for an invented LTX bridge.");
  }

  function updatePhoto(index: number, field: "area" | "viewpoint", value: string) {
    setPhotos((current) => current.map((photo, i) => i === index ? { ...photo, [field]: value } : photo));
  }

  function addBridge() {
    if (bridgeFrom === bridgeTo || bridgePairs.has(`${bridgeFrom}-${bridgeTo}`)) return;
    setBridges((current) => [...current, { from_index: bridgeFrom, to_index: bridgeTo, kind: bridgeKind, duration_seconds: bridgeDuration }]);
  }

  function photoLabel(index: number) {
    const photo = photos[index];
    return `${String(index + 1).padStart(2, "0")} · ${photo?.area ?? "unclassified"} · ${photo?.viewpoint ?? "unclassified"}`;
  }

  async function upload(files: File[]) {
    const form = new FormData(); files.forEach((file) => form.append("files", file));
    const response = await fetch(api("/uploads"), { method: "POST", body: form });
    const body = await response.json(); if (!response.ok) throw new Error(body.detail ?? "Upload failed.");
    return body.source_paths as string[];
  }

  async function waitForJob(id: string) {
    let job: { status: string; error?: string; result?: { qa?: Qa[] } };
    do { await new Promise((resolve) => setTimeout(resolve, 750)); job = await fetch(api(`/jobs/${id}`)).then((r) => r.json()); setStatus(`${job.status}…`); } while (["queued", "running"].includes(job.status));
    if (job.status !== "succeeded") throw new Error(job.error ?? "The local job failed.");
    setQa(job.result?.qa ?? []); return job;
  }

  async function generateReel() {
    if (photos.length < 2) return;
    setStatus("Uploading property views…"); setVideo(undefined);
    try {
      const source_paths = await upload(photos.map((photo) => photo.file));
      setStatus("Building reel and evaluating input quality…");
      const response = await fetch(api("/jobs"), { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({
        kind: "fixture_reel", source_paths, runtime_profile_name: profile,
        settings: { target_duration_seconds: duration, transition_seconds: dissolve },
        image_annotations: photos.map((photo, index) => ({ index, area: photo.area, viewpoint: photo.viewpoint })), invented_bridges: bridges,
      }) });
      const job = await response.json(); if (!response.ok) throw new Error(job.detail ?? "Could not start the reel.");
      await waitForJob(job.id);
      const artifacts = await fetch(api(`/jobs/${job.id}/artifacts`)).then((r) => r.json());
      const mp4 = artifacts.find((artifact: { name: string }) => artifact.name.endsWith(".mp4"));
      setVideo(api(`/jobs/${job.id}/artifacts/${mp4.name}`)); setStatus("Reel ready. Review its QA before delivery."); setTab("quality");
    } catch (error) { setStatus(error instanceof Error ? error.message : "Could not generate reel."); }
  }

  async function editPhoto() {
    const source = photos[editSource];
    if (!source) return; setEditing(true); setStatus("Uploading and checking the source photo…");
    try {
      const [source_path] = await upload([source.file]);
      const response = await fetch(api("/jobs"), { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ kind: "image_edit", source_paths: [source_path], runtime_profile_name: profile, instruction: editPrompt }) });
      const job = await response.json(); if (!response.ok) throw new Error(job.detail ?? "Could not start the edit.");
      await waitForJob(job.id); setStatus("Edit candidates are ready in the local job artifacts."); setTab("quality");
    } catch (error) { setStatus(error instanceof Error ? error.message : "Could not create edit candidates."); } finally { setEditing(false); }
  }

  async function checkReadiness() {
    if (!loraDataset || !loraEvidence) return;
    setStatus("Checking whether your training data is ready…");
    try {
      const [dataset, evidence] = await Promise.all([loraDataset.text(), loraEvidence.text()]);
      const response = await fetch(api("/lora/readiness"), {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ dataset: JSON.parse(dataset), evidence: JSON.parse(evidence) }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail ?? "Those files are not valid readiness manifests.");
      setLoraResult(result); setStatus(result.training_permitted ? "Your data is ready for a reviewed pilot." : "Training is not justified yet. See what to prepare below.");
    } catch (error) { setStatus(error instanceof Error ? error.message : "Could not check readiness."); }
  }

  function explainReason(reason: string) {
    const explanations: Record<string, string> = {
      asset_rights_incomplete: "Some images do not record permission for derivative, training, and portfolio use.",
      pair_not_same_camera_view: "A daylight and dusk pair must use the same camera view.",
      pair_geometry_unverified: "Confirm that each matched pair preserves the same visible building geometry.",
      property_split_leakage: "Keep every photo of one property in exactly one split: train, validation, or held-out test.",
      missing_train_pairs: "Add at least one licensed training pair.",
      missing_validation_pairs: "Add a separate property for validation.",
      missing_test_pairs: "Add a separate held-out property for final evaluation.",
    };
    return explanations[reason] ?? `Missing evidence: ${reason.replace(/^missing_/, "").replaceAll("_", " ")}.`;
  }

  return <main>
    <header className="topbar"><a className="brand" href="#studio"><img src="/static/reelmelisting-logo.png" alt="ReelMeListing" /><span>ReelMe<span>Listing</span></span></a><div className="machine"><i className={selectedRuntime?.compatible ? "online" : ""} /> {selectedRuntime?.device.toUpperCase() ?? "LOCAL"} · {selectedRuntime?.compatible ? "ready" : selectedRuntime?.warning ?? "checking"}</div></header>
    <section className="hero"><p className="kicker">LOCAL ARCHITECTURAL STUDIO</p><h1>Make the property<br /><em>move with purpose.</em></h1><p>Source-backed reels, deliberate invented bridges, real QA evidence, and local model control.</p></section>
    <nav className="tabs" aria-label="Studio sections">{(["studio", "edit", "quality", "training"] as Tab[]).map((item) => <button key={item} className={tab === item ? "active" : ""} onClick={() => setTab(item)}>{item === "studio" ? "Reel studio" : item === "edit" ? "Image edit" : item === "quality" ? "Quality" : "Improve a treatment"}</button>)}</nav>
    {tab === "studio" && <section className="panel studio" id="studio"><div className="panel-head"><div><p className="kicker">01 / PROPERTY STORYBOARD</p><h2>Upload and label every view.</h2></div><label className="upload">Add photos<input type="file" accept="image/*" multiple onChange={(event) => addPhotos(event.target.files)} /></label></div>
      {photos.length === 0 ? <div className="empty"><span>✦</span><strong>Start with 2–12 photos of one property.</strong><p>Keep unrelated properties in separate projects.</p></div> : <div className="photo-grid">{photos.map((photo, index) => <article className="photo-card" key={photo.url}><img src={photo.url} alt={`Property view ${index + 1}`} /><div className="photo-meta"><b>{String(index + 1).padStart(2, "0")}</b><select value={photo.area} onChange={(event) => updatePhoto(index, "area", event.target.value)}><option>unclassified</option><option>front</option><option>backyard</option><option>patio</option><option>pool</option><option>detail</option></select><select value={photo.viewpoint} onChange={(event) => updatePhoto(index, "viewpoint", event.target.value)}><option>unclassified</option><option>left angle</option><option>wide view</option><option>right angle</option><option>close detail</option><option>same framing</option></select></div></article>)}</div>}
      {photos.length > 1 && <section className="bridge-workflow"><aside className="bridge-guide"><p className="kicker">OPTIONAL · LTX CAMERA BRIDGE</p><h3>Connect compatible views.</h3><p>Use an invented bridge only when the camera could plausibly move from one view to the next.</p><ul><li><b>Camera move:</b> same or adjacent area, different angle.</li><li><b>Lighting only:</b> same framing, different time of day.</li><li><b>Use a cut:</b> unrelated areas or uncertain geometry.</li></ul><small>Synthetic visualization—not a verified walkthrough.</small></aside><div className="bridge-setup"><div className="bridge-setup-head"><div><b>Create a bridge</b><span>It will be generated and checked by LTX QA.</span></div><span>{bridges.length} planned</span></div><div className="bridge-form"><label>Start view<select value={bridgeFrom} onChange={(event) => setBridgeFrom(+event.target.value)}>{photos.map((_, index) => <option key={index} value={index}>{photoLabel(index)}</option>)}</select></label><label>End view<select value={bridgeTo} onChange={(event) => setBridgeTo(+event.target.value)}>{photos.map((_, index) => <option key={index} value={index}>{photoLabel(index)}</option>)}</select></label><fieldset><legend>Transition intent</legend><label><input type="radio" checked={bridgeKind === "spatial"} onChange={() => setBridgeKind("spatial")} /> Camera move</label><label><input type="radio" checked={bridgeKind === "lighting"} onChange={() => setBridgeKind("lighting")} /> Lighting only</label></fieldset><label>Length <output>{bridgeDuration.toFixed(1)}s</output><input type="range" min="0.75" max="5" step="0.25" value={bridgeDuration} onChange={(event) => setBridgeDuration(+event.target.value)} /></label><button disabled={bridgeFrom === bridgeTo || bridgePairs.has(`${bridgeFrom}-${bridgeTo}`)} onClick={addBridge}>Plan bridge</button></div>{bridges.length > 0 && <div className="bridge-list">{bridges.map((bridge, index) => <article key={`${bridge.from_index}-${bridge.to_index}`}><span>{String(index + 1).padStart(2, "0")}</span><div><b>{photoLabel(bridge.from_index)} → {photoLabel(bridge.to_index)}</b><p>{bridge.kind === "spatial" ? "Invented camera move" : "Lighting-only transition"} · {bridge.duration_seconds.toFixed(1)} seconds</p></div><button onClick={() => setBridges((current) => current.filter((item) => item !== bridge))}>Remove</button></article>)}</div>}</div></section>}
      <div className="controls"><label>Final length <output>{duration}s</output><input type="range" min="8" max="20" value={duration} onChange={(event) => setDuration(+event.target.value)} /></label><label>Scene dissolve <output>{dissolve.toFixed(1)}s</output><input type="range" min="0.2" max="5" step="0.1" value={dissolve} onChange={(event) => setDissolve(+event.target.value)} /></label><label>Render profile <select value={profile} onChange={(event) => setProfile(event.target.value)}><option value="local_mps">M5 preview</option><option value="remote_cuda">CUDA / RTX</option></select><small>{selectedRuntime?.warning ?? "Model status is checked locally."}</small></label></div>
      <button className="primary" disabled={photos.length < 2} onClick={generateReel}>Generate reel <span>→</span></button></section>}
    {tab === "edit" && <section className="panel editor"><p className="kicker">02 / IMAGE EDITING</p><h2>Set the daylight-to-dusk treatment.</h2><p>The InstructPix2Pix editor uses the selected local runtime. A photo must pass input QA before it can generate candidates.</p><div className="edit-source"><label>Source photo <select value={editSource} onChange={(event) => setEditSource(+event.target.value)} disabled={!photos.length}>{photos.map((photo, index) => <option key={photo.url} value={index}>{String(index + 1).padStart(2, "0")} · {photo.file.name}</option>)}</select></label><label className="upload secondary">Add a photo to storyboard<input type="file" accept="image/*" onChange={(event) => addPhotos(event.target.files)} /></label>{photos[editSource] && <img src={photos[editSource].url} alt="Selected edit source" />}</div><textarea value={editPrompt} onChange={(event) => setEditPrompt(event.target.value)} /><div className="edit-actions"><span>{photos[editSource] ? `Editing: ${photos[editSource].file.name}` : "Upload a source here or add photos in Reel studio first."}</span><button className="primary" disabled={!photos[editSource] || editing} onClick={editPhoto}>{editing ? "Generating…" : "Generate edit candidates"} <span>→</span></button></div></section>}
    {tab === "quality" && <section className="panel quality"><p className="kicker">03 / QUALITY EVIDENCE</p><h2>Review what the pipeline measured.</h2><p>Quality gates inform a human delivery decision; they do not certify property truthfulness.</p>{qa.length === 0 ? <div className="empty"><span>✓</span><strong>No run selected yet.</strong><p>Generate a reel or edit photo to see its per-image QA results here.</p></div> : <div className="qa-list">{qa.map((item) => <article className={`qa ${item.decision}`} key={item.source_path}><span>{item.decision}</span><div><b>{item.source_path.split("/").pop()}</b><p>{item.reason_codes.length ? item.reason_codes.join(" · ") : "No automated warnings"}</p></div></article>)}</div>}{video && <div className="player"><video src={video} controls playsInline /><a href={video} download>Save MP4</a></div>}</section>}
    {tab === "training" && <section className="panel training"><p className="kicker">OPTIONAL · STYLE IMPROVEMENT</p><h2>Teach one treatment,<br />not a whole new model.</h2><p>A LoRA is a small add-on trained on your own approved examples. Here, it could make the day-to-dusk treatment more consistent for exterior property photos while the base image model stays frozen.</p><div className="training-callout"><b>Most users do not need this yet.</b><span>First use the normal Image Edit workflow. Consider a LoRA only after you have many licensed before/after examples and your baseline QA is reliable.</span></div><div className="learning-cards"><article><span>01</span><h3>What it learns</h3><p>Your narrow visual treatment: natural warm dusk, controlled skies, and geometry preservation.</p></article><article><span>02</span><h3>What it cannot promise</h3><p>It cannot verify a property, invent unseen features safely, or train the video model.</p></article><article><span>03</span><h3>What you must bring</h3><p>Licensed daytime → dusk pairs of the same property and same camera view, plus held-out properties for testing.</p></article></div><section className="readiness"><div><p className="kicker">READY WHEN YOU HAVE</p><h3>Four things before a pilot.</h3><ol><li><b>Rights recorded</b><span>Every image allows derivative, training, and portfolio use.</span></li><li><b>Real matched pairs</b><span>The same house and camera position in daylight and dusk.</span></li><li><b>A fair test</b><span>Entire properties separated into train, validation, and held-out test sets.</span></li><li><b>Pipeline evidence</b><span>Geometry QA, conditioning comparison, video QA, and baseline results.</span></li></ol></div><div className="readiness-check"><p className="kicker">CHECK YOUR PREPARATION</p><h3>Already have the manifests?</h3><p>Upload the local JSON files. This check does not upload your photos or start training.</p><label>Dataset manifest<input type="file" accept="application/json" onChange={(event) => setLoraDataset(event.target.files?.[0])} /></label><label>Evidence manifest<input type="file" accept="application/json" onChange={(event) => setLoraEvidence(event.target.files?.[0])} /></label><button className="primary" disabled={!loraDataset || !loraEvidence} onClick={checkReadiness}>Check readiness <span>→</span></button></div></section>{loraResult && <section className={`readiness-result ${loraResult.training_permitted ? "ready" : "not-ready"}`}><b>{loraResult.training_permitted ? "Ready for a reviewed LoRA pilot" : "Not ready to train yet"}</b><p>{loraResult.training_permitted ? `You have ${loraResult.pair_counts.train} training, ${loraResult.pair_counts.validation} validation, and ${loraResult.pair_counts.test} held-out pairs. Review the plan before starting a CUDA pilot.` : "Here is the specific preparation still missing:"}</p>{!loraResult.training_permitted && <ul>{loraResult.reason_codes.map((reason) => <li key={reason}>{explainReason(reason)}</li>)}</ul>}</section>}</section>}
    <footer><span>{status}</span><span>Local-first · synthetic bridges labeled in lineage</span></footer>
  </main>;
}

createRoot(document.getElementById("root")!).render(<App />);
