import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { trpc } from "@/lib/trpc";
import { AlertCircle, ArrowLeft, FileImage, Loader2, LockKeyhole, UploadCloud } from "lucide-react";
import { ChangeEvent, FormEvent, useState } from "react";
import { Link, useLocation } from "wouter";

export default function NewScreening() {
  const [, setLocation] = useLocation();
  const [patientId, setPatientId] = useState("");
  const [patientAge, setPatientAge] = useState("");
  const [diabetesDuration, setDiabetesDuration] = useState("");
  const [hba1c, setHba1c] = useState("");
  const [systolic, setSystolic] = useState("");
  const [clinicalQuestion, setClinicalQuestion] = useState("");
  const [imageName, setImageName] = useState("");
  const [imageData, setImageData] = useState("");
  const [preview, setPreview] = useState("");
  const [formError, setFormError] = useState("");
  const createCase = trpc.screening.create.useMutation({ onSuccess: result => { if (result) setLocation(`/app/cases/${result.screeningCase.id}`); } });

  const handleFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setFormError("");
    if (!['image/jpeg', 'image/png'].includes(file.type) || file.size > 6 * 1024 * 1024) { setFormError("Use a JPEG or PNG retinal image smaller than 6 MB."); return; }
    const reader = new FileReader();
    reader.onload = () => { const result = String(reader.result); setImageData(result); setPreview(result); setImageName(file.name); };
    reader.onerror = () => setFormError("The selected image could not be read.");
    reader.readAsDataURL(file);
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    setFormError("");
    if (!patientId.trim() || !patientAge || !imageData) { setFormError("Patient identifier, age, and a retinal image are required."); return; }
    createCase.mutate({ patientId: patientId.trim(), patientAge: Number(patientAge), diabetesDurationYears: diabetesDuration ? Number(diabetesDuration) : undefined, hba1c: hba1c ? Number(hba1c) : undefined, systolicBloodPressure: systolic ? Number(systolic) : undefined, clinicalQuestion: clinicalQuestion.trim() || undefined, imageName, imageData });
  };

  return <div className="mx-auto max-w-5xl"><Link href="/app" className="no-print inline-flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground"><ArrowLeft className="h-4 w-4" />Back to workspace</Link><div className="mt-6 grid gap-8 lg:grid-cols-[1fr_0.82fr]"><section><p className="technical-label text-primary">case intake / structured input</p><h1 className="mt-2 text-4xl font-bold tracking-[-0.055em]">Start a new screening.</h1><p className="mt-3 max-w-xl leading-7 text-muted-foreground">The retinal image is kept as the focal input. Patient variables provide clinical context for the future risk-estimation pipeline.</p><form onSubmit={submit} className="blueprint-card mt-8 rounded-2xl p-5 sm:p-7"><div className="grid gap-5 sm:grid-cols-2"><Field label="Patient identifier" required><Input value={patientId} onChange={e => setPatientId(e.target.value)} placeholder="e.g. PT-204" /></Field><Field label="Age" required><Input value={patientAge} onChange={e => setPatientAge(e.target.value)} type="number" min="18" max="120" placeholder="58" /></Field><Field label="Diabetes duration (years)"><Input value={diabetesDuration} onChange={e => setDiabetesDuration(e.target.value)} type="number" min="0" step="0.1" placeholder="12" /></Field><Field label="HbA1c (%)"><Input value={hba1c} onChange={e => setHba1c(e.target.value)} type="number" min="0" step="0.1" placeholder="7.4" /></Field><Field label="Systolic blood pressure"><Input value={systolic} onChange={e => setSystolic(e.target.value)} type="number" min="60" max="260" placeholder="132" /></Field></div><div className="mt-5"><Field label="Clinical question"><Textarea value={clinicalQuestion} onChange={e => setClinicalQuestion(e.target.value)} className="min-h-24 resize-none" placeholder="Optional: What context should the evidence review address?" /></Field></div><div className="mt-7 flex flex-col-reverse gap-3 border-t border-border pt-5 sm:flex-row sm:justify-end"><Link href="/app"><Button type="button" variant="ghost">Cancel</Button></Link><Button disabled={createCase.isPending} className="button-press min-w-44"><>{createCase.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}Create case record</></Button></div>{(formError || createCase.error) && <div className="mt-5 flex gap-3 rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-900"><AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />{formError || createCase.error?.message}</div>}</form></section><aside className="space-y-4"><div className="blueprint-card rounded-2xl p-5"><p className="technical-label text-primary">retinal image</p><label className="mt-4 grid min-h-72 place-items-center overflow-hidden rounded-xl border border-dashed border-primary/35 bg-secondary/35 p-5 text-center transition-colors hover:bg-secondary/60">{preview ? <img src={preview} alt="Selected retinal image preview" className="max-h-64 w-full rounded-lg object-contain" /> : <span><span className="mx-auto grid h-12 w-12 place-items-center rounded-full border border-primary/30 bg-background"><UploadCloud className="h-5 w-5 text-primary" /></span><span className="mt-4 block text-sm font-semibold">Upload retinal image</span><span className="mt-1 block text-xs leading-5 text-muted-foreground">JPEG or PNG · maximum 6 MB</span></span>}<input type="file" accept="image/jpeg,image/png" onChange={handleFile} className="sr-only" /></label>{imageName && <p className="mt-3 flex items-center gap-2 font-mono text-xs text-muted-foreground"><FileImage className="h-3.5 w-3.5 text-primary" />{imageName}</p>}</div><div className="rounded-2xl border border-primary/20 bg-cyan-50/70 p-5"><div className="flex gap-3"><LockKeyhole className="mt-0.5 h-4 w-4 shrink-0 text-primary" /><div><p className="text-sm font-semibold">Protected, traceable input</p><p className="mt-1.5 text-xs leading-5 text-muted-foreground">The case is stored in your authenticated workspace. Results remain explicitly marked as pending until a validated model pipeline returns them.</p></div></div></div></aside></div></div>;
}

function Field({ label, required, children }: { label: string; required?: boolean; children: React.ReactNode }) { return <div className="space-y-2"><Label className="text-sm font-semibold">{label}{required && <span className="ml-1 text-primary">*</span>}</Label>{children}</div>; }
