import { Button } from "@/components/ui/button";
import { startLogin } from "@/const";
import { Activity, ArrowRight, BookOpenCheck, BrainCircuit, Database, Eye, FileSearch, ShieldCheck } from "lucide-react";
import { Link } from "wouter";

const modules = [
  { icon: Eye, code: "01 / VISUAL", title: "Retinal image review", text: "Capture a retinal image and preserve a focused case record for model-assisted visual assessment." },
  { icon: Activity, code: "02 / CLINICAL", title: "Risk context", text: "Structure key clinical variables so risk-model outputs can be reviewed alongside image findings." },
  { icon: BookOpenCheck, code: "03 / EVIDENCE", title: "Source-grounded review", text: "Keep retrieved guidelines and literature separate, readable, and traceable to their sources." },
  { icon: BrainCircuit, code: "04 / SYNTHESIS", title: "Clear explanation", text: "Bring model outputs and supporting evidence together for professional review—not autonomous diagnosis." },
];

export default function Home() {
  return (
    <div className="min-h-screen overflow-hidden">
      <header className="container no-print flex h-20 items-center justify-between gap-4">
        <Link href="/" className="flex items-center gap-3"><SignalMark /><span><span className="technical-label block text-primary">clinical intelligence</span><span className="text-sm font-bold tracking-tight">Retina Signal</span></span></Link>
        <nav className="hidden items-center gap-7 text-sm font-medium text-muted-foreground md:flex"><a href="#workflow" className="hover:text-foreground">Workflow</a><a href="#architecture" className="hover:text-foreground">Architecture</a><a href="#safety" className="hover:text-foreground">Scope</a></nav>
        <Button onClick={() => startLogin()} variant="outline" className="button-press border-primary/30 bg-background/60 text-xs sm:text-sm">Open workspace <ArrowRight className="ml-2 h-4 w-4" /></Button>
      </header>

      <main>
        <section className="container relative pb-20 pt-12 sm:pb-28 sm:pt-20">
          <div className="pointer-events-none absolute right-[-8rem] top-[-5rem] hidden h-[32rem] w-[32rem] md:block"><div className="wireframe-orbit absolute inset-12" /><div className="wireframe-orbit-pink absolute inset-0" /><div className="dot-field absolute left-16 top-16 h-36 w-36 rounded-full opacity-60" /><RetinaScanDiagram /><span className="technical-label absolute bottom-10 right-4 text-pink-500/70">f(x) = evidence + signal</span></div>
          <div className="relative max-w-4xl">
            <p className="technical-label inline-flex items-center gap-2 rounded-full border border-primary/25 bg-secondary/70 px-3 py-1.5 text-primary"><span className="h-1.5 w-1.5 rounded-full bg-primary" />research clinical decision support</p>
            <h1 className="mt-7 max-w-4xl text-5xl font-bold leading-[0.96] tracking-[-0.052em] text-foreground sm:text-7xl lg:text-8xl">Make retinal screening <span className="relative whitespace-nowrap">legible.<span className="absolute -bottom-2 left-1/2 h-2 w-[105%] -translate-x-1/2 -rotate-1 bg-pink-200/75" /></span></h1>
            <p className="mt-8 max-w-2xl text-lg leading-8 text-muted-foreground sm:text-xl">A structured workspace that connects retinal image intake, clinical risk context, medical evidence, and explainable model outputs for qualified professional review.</p>
            <div className="mt-9 flex flex-col gap-3 sm:flex-row"><Button onClick={() => startLogin()} size="lg" className="button-press h-12 rounded-xl px-6 text-sm shadow-lg shadow-primary/15">Start a screening <ArrowRight className="ml-2 h-4 w-4" /></Button><a href="#workflow"><Button variant="outline" size="lg" className="button-press h-12 w-full rounded-xl border-primary/25 bg-background/70 px-6 text-sm sm:w-auto">Explore the workflow</Button></a></div>
          </div>
          <div className="relative mt-16 grid gap-3 sm:grid-cols-3"><div className="blueprint-card surface-hover rounded-2xl p-5 sm:col-span-2"><p className="technical-label text-primary">signal fusion / 04 layers</p><div className="mt-6 flex items-center gap-2"><div className="h-2 flex-1 rounded-full bg-cyan-200" /><div className="h-2 flex-1 rounded-full bg-pink-200" /><div className="h-2 flex-1 rounded-full bg-cyan-300" /><div className="h-2 flex-1 rounded-full bg-slate-200" /></div><p className="mt-5 text-sm font-medium leading-6">Visual signal + clinical context + retrieved evidence + grounded explanation</p></div><div className="blueprint-card surface-hover rounded-2xl p-5"><p className="technical-label text-muted-foreground">prototype guardrail</p><p className="mt-3 text-xl font-semibold tracking-tight">Assist review.<br />Never replace it.</p></div></div>
        </section>

        <section id="workflow" className="border-y border-border bg-background/65 py-20 sm:py-24"><div className="container"><div className="max-w-2xl"><p className="technical-label text-primary">the screening workflow</p><h2 className="mt-3 text-3xl font-bold tracking-[-0.045em] sm:text-5xl">One case. Four distinct kinds of evidence.</h2><p className="mt-5 text-lg leading-7 text-muted-foreground">Each layer retains its own identity so a clinician can distinguish patient data, model findings, retrieved medical evidence, and the final explanatory summary.</p></div><div className="mt-12 grid gap-4 md:grid-cols-2">{modules.map((module, index) => <article key={module.code} className="blueprint-card surface-hover group relative overflow-hidden rounded-2xl p-6"><span className="absolute -right-6 -top-6 grid h-28 w-28 place-items-center rounded-full border border-primary/15 text-4xl font-bold text-primary/10">0{index + 1}</span><module.icon className="h-5 w-5 text-primary" /><p className="technical-label mt-7 text-muted-foreground">{module.code}</p><h3 className="mt-2 text-2xl font-bold tracking-tight">{module.title}</h3><p className="mt-3 max-w-md text-sm leading-6 text-muted-foreground">{module.text}</p></article>)}</div></div></section>

        <section id="architecture" className="container py-20 sm:py-28"><div className="grid gap-10 lg:grid-cols-[0.82fr_1.18fr] lg:items-center"><div><p className="technical-label text-primary">architecture / traceable by design</p><h2 className="mt-3 text-4xl font-bold tracking-[-0.05em] sm:text-5xl">A composed system—not a black box.</h2><p className="mt-6 leading-7 text-muted-foreground">Retina Signal is structured around three intelligence pipelines: computer vision for image analysis, clinical machine learning for risk estimation, and retrieval for source-grounded knowledge. The workspace keeps their outputs visible before any explanation layer is shown.</p></div><div className="blueprint-card relative overflow-hidden rounded-3xl p-6 sm:p-9"><div className="absolute inset-0 opacity-50" style={{ backgroundImage: "linear-gradient(135deg, transparent 49.7%, rgba(117, 205, 220, .35) 50%, transparent 50.3%)", backgroundSize: "76px 76px" }} /><div className="relative grid gap-3"><ArchitectureRow icon={Database} code="INPUT" text="Retinal image + structured patient variables" /><ArchitectureRow icon={Eye} code="VISION" text="Severity and optional visual findings" /><ArchitectureRow icon={FileSearch} code="RETRIEVAL" text="Relevant guidelines and research sources" /><ArchitectureRow icon={BrainCircuit} code="FUSION" text="Separated inputs for a grounded explanation" /></div></div></div></section>

        <section id="safety" className="border-t border-border bg-slate-950 py-16 text-slate-100"><div className="container grid gap-8 md:grid-cols-[auto_1fr]"><div className="grid h-12 w-12 place-items-center rounded-xl border border-cyan-300/30 bg-cyan-300/10"><ShieldCheck className="h-6 w-6 text-cyan-200" /></div><div><p className="technical-label text-cyan-200">scope and limitation</p><h2 className="mt-2 text-2xl font-bold tracking-tight">Designed to support qualified clinical review.</h2><p className="mt-3 max-w-3xl text-sm leading-7 text-slate-300">This research prototype organizes inputs, model outputs, and evidence for professional interpretation. It does not autonomously diagnose diabetic retinopathy, provide treatment recommendations, or replace clinical judgment.</p></div></div></section>
      </main>
      <footer className="container flex flex-col gap-3 py-8 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between"><span className="flex items-center gap-2 technical-label"><SignalMark compact />RETINA SIGNAL / RESEARCH PROTOTYPE</span><span>Built around traceability, uncertainty, and clinical review.</span></footer>
    </div>
  );
}

function ArchitectureRow({ icon: Icon, code, text }: { icon: typeof Database; code: string; text: string }) {
  return <div className="flex items-center gap-4 rounded-xl border border-border bg-background/80 p-4"><span className="grid h-9 w-9 place-items-center rounded-lg bg-secondary text-primary"><Icon className="h-4 w-4" /></span><div><p className="technical-label text-muted-foreground">{code}</p><p className="mt-0.5 text-sm font-semibold">{text}</p></div></div>;
}

function SignalMark({ compact = false }: { compact?: boolean }) {
  return <span aria-hidden="true" className={`relative grid shrink-0 place-items-center rounded-full border border-primary/45 bg-cyan-50 ${compact ? "h-5 w-5" : "h-9 w-9"}`}><span className={`absolute rounded-full border border-primary/45 ${compact ? "inset-1" : "inset-1.5"}`} /><span className={`absolute rounded-full border border-pink-400/60 ${compact ? "inset-2" : "inset-3"}`} /><span className={`rounded-full bg-primary ${compact ? "h-1 w-1" : "h-1.5 w-1.5"}`} /><span className={`absolute rounded-full bg-pink-400 ${compact ? "right-0.5 top-0.5 h-1 w-1" : "right-1 top-1 h-1.5 w-1.5"}`} /></span>;
}

function RetinaScanDiagram() {
  return <svg viewBox="0 0 520 520" className="absolute inset-0 h-full w-full opacity-70" aria-hidden="true"><circle cx="260" cy="260" r="132" fill="none" stroke="rgba(28,126,144,.24)" strokeWidth="1" /><circle cx="308" cy="224" r="33" fill="rgba(245,182,211,.16)" stroke="rgba(218,117,167,.45)" strokeWidth="1" /><path d="M306 253C273 275 247 302 220 343M303 253C337 276 372 299 408 333M285 242C242 228 202 204 165 170M293 226C266 191 239 156 216 128M321 238C365 211 400 184 432 151" fill="none" stroke="rgba(26,137,151,.54)" strokeWidth="1.25" strokeLinecap="round" /><path d="M220 343l-12 23m12-23 27 3m161-13 18 23m-18-23-25 0m-218-163-16-22m16 22-26 4m77-42-2-25m2 25-24-8m234 23 23-13m-23 13 3 24" fill="none" stroke="rgba(218,117,167,.48)" strokeWidth="1" /><text x="113" y="403" fill="rgba(26,137,151,.68)" fontSize="11" fontFamily="IBM Plex Mono, monospace">RETINAL FIELD / SIGNAL MAP</text><text x="340" y="390" fill="rgba(218,117,167,.68)" fontSize="11" fontFamily="IBM Plex Mono, monospace">Δx · Δy · θ</text></svg>;
}
