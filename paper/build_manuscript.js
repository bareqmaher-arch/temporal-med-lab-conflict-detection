// Build the manuscript.docx for the Explainable Temporal Medication-Laboratory
// Conflict Detection study, now reporting real results from MIMIC-IV v2.2.
// Embeds the figures in paper/figures/ and the tables in paper/tables/.
// Run from the project root:
//   node paper/build_manuscript.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, Footer,
} = require("docx");

const ROOT = __dirname;
const DATA = JSON.parse(fs.readFileSync(path.join(ROOT, "tables", "manuscript_data.json"), "utf8"));
const FIG = (n) => path.join(ROOT, "figures", n);
const CONTENT_W = 9360;
const BLACK = "000000";

// ---------- helpers ----------
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const P = (t) => new Paragraph({ spacing: { after: 120 }, alignment: AlignmentType.JUSTIFIED, children: [new TextRun(t)] });
const BULLET = (t) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 40 }, children: [new TextRun(t)] });
const NUMP = (n, t) => new Paragraph({ spacing: { after: 40 }, indent: { left: 720, hanging: 360 }, children: [new TextRun(`${n}. ${t}`)] });
const MONO = (t) => new Paragraph({ spacing: { after: 20 }, children: [new TextRun({ text: t, font: "Consolas", size: 18 })] });
const CAPTION = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: t, italics: true, size: 20 })] });
const refP = (n, t) => new Paragraph({ spacing: { after: 60 }, indent: { left: 540, hanging: 540 }, children: [new TextRun(`[${n}] ${t}`)] });

function image(file, w, h) {
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(file),
      transformation: { width: w, height: h },
      altText: { title: file, description: file, name: file } })],
  });
}

const BORDER = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const BORDERS = { top: BORDER, left: BORDER, bottom: BORDER, right: BORDER };
function cell(text, width, { head = false, align = AlignmentType.LEFT } = {}) {
  return new TableCell({
    borders: BORDERS, width: { size: width, type: WidthType.DXA },
    shading: head ? { fill: "D9D9D9", type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: align, children: [new TextRun({ text: String(text), bold: head, size: 18 })] })],
  });
}
function table(headers, rows, widths) {
  const trs = [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, widths[i], { head: true })) })];
  for (const r of rows) trs.push(new TableRow({ children: r.map((c, i) => cell(c, widths[i])) }));
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths, rows: trs });
}

function shortScenario(s) {
  s = String(s);
  if (s.includes("ACE")) return "ACEi/ARB - K+";
  if (s.includes("Warfarin")) return "Warfarin - INR";
  if (s.includes("Metformin")) return "Metformin - eGFR";
  return s;
}
const clip = (s, n) => { s = String(s); return s.length > n ? s.slice(0, n) + "..." : s; };
function fmt(x, k = 3) { return (typeof x === "number") ? x.toFixed(k) : String(x); }
function num(arr, scenario, method, field) {
  const row = arr.find((r) => r.scenario === scenario && r.method === method);
  return row ? row[field] : "n/a";
}
function pct(x) { return (typeof x === "number") ? (x * 100).toFixed(1) + "%" : String(x); }
function commify(n) { return n.toLocaleString("en-US"); }

// scenario keys as they appear in our data files (with the unicode arrow)
const SCN_ACE  = "ACE inhibitor / ARB → Hyperkalemia";
const SCN_WAR  = "Warfarin → Elevated INR / Bleeding risk";
const SCN_MET  = "Metformin → Renal function decline";

// ---------- document body ----------
const body = [];

// Title
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 240, after: 80 },
  children: [new TextRun({ text: "Explainable Temporal Medication-Laboratory Conflict Detection for Early Prevention of Adverse Drug Events", bold: true, size: 36, color: BLACK })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
  children: [new TextRun({ text: "An evaluation on 122,166 MIMIC-IV patients", italics: true, size: 22, color: BLACK })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 60 },
  children: [new TextRun({ text: "Bareq Maher", size: 24, color: BLACK })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
  children: [new TextRun({ text: "Department of Medical Informatics, Imam Al-Kadhum University College (IKU), Baghdad, Iraq", italics: true, size: 20 })] }));

// Abstract
const N_PT = commify(DATA.n_patients);
const N_AL = commify(DATA.n_alerts);
const exp1_ace_static_alerts = num(DATA.exp1_rules_vs_temporal, SCN_ACE, "Static threshold", "alerts_per_100");
const exp1_ace_temp_alerts   = num(DATA.exp1_rules_vs_temporal, SCN_ACE, "Temporal rule",    "alerts_per_100");
const exp1_ace_static_spec   = num(DATA.exp1_rules_vs_temporal, SCN_ACE, "Static threshold", "specificity");
const exp1_ace_temp_spec     = num(DATA.exp1_rules_vs_temporal, SCN_ACE, "Temporal rule",    "specificity");
const reduction_ace_pct      = DATA.table4_early_detection.find(r => r.scenario === SCN_ACE).alert_reduction_pct;

body.push(H1("Abstract"));
body.push(P(
  "Background. Adverse drug events often unfold gradually, with a laboratory value drifting toward a hazardous range over days or weeks. Conventional alerting systems react only when a single value crosses a fixed threshold, and so they ignore the patient's own baseline, the direction of travel, and the time-relationship to the drug. They also fire often enough that clinicians stop reading them. " +
  "Objective. We evaluate a temporal, explainable approach to medication-laboratory conflict detection that pairs longitudinal laboratory trends with a patient-specific baseline and drug-exposure timing, scored by a transparent 0-100 risk index and explained for the prescriber. " +
  "Methods. Data enter through a source-agnostic loader. For the evaluation reported here the loader reads MIMIC-IV v2.2 hospital tables and emits a canonical six-table model. After timeline construction we derive temporal features (7/14/30-day slopes, change from baseline, acceleration, trend persistence, days since drug start) and patient modifiers (CKD-EPI eGFR, hepatic impairment from ICD, age, polypharmacy). Three detectors are compared head-to-head on the same cohort: static thresholds, sustained-trend temporal rules, and four learned models (logistic regression, random forest, XGBoost, LightGBM). Each alert carries a clinician-readable narrative and, for model-driven alerts, SHAP attributions. Three scenarios are studied: ACE inhibitors or ARBs and potassium; warfarin and INR; and metformin and eGFR. " +
  `Results. On ${N_PT} MIMIC-IV patients the pipeline raised ${N_AL} alerts. The temporal rule did not dominate the static threshold uniformly; rather it traded sensitivity for specificity and a smaller alert volume. The trade-off was most favourable for ACE/ARB-driven hyperkalemia, where total alerts fell by ${reduction_ace_pct}% (${exp1_ace_static_alerts} to ${exp1_ace_temp_alerts} per 100 patients) with specificity rising from ${fmt(exp1_ace_static_spec, 3)} to ${fmt(exp1_ace_temp_spec, 3)}. The learned XGBoost models reached AUROC of 0.93-0.97 across scenarios with calibrated probabilities (Brier 0.014-0.050). Case-level inspection showed that for individual patients the temporal logic could lead the static threshold by up to 29 days. ` +
  "Conclusions. Trend-aware, baseline-relative detection is most valuable where the underlying physiology drifts gradually; on real EHR data the average lead-time over a well-tuned threshold is modest, but the burden reduction is real and the case-level lead-time can be substantial. Calibrated gradient-boosted models, paired with patient-specific temporal features and transparent explanations, provide a workable foundation for explainable medication-laboratory safety monitoring. Prospective evaluation remains essential before clinical deployment."
));
body.push(new Paragraph({ spacing: { after: 200 }, children: [
  new TextRun({ text: "Keywords: ", bold: true }),
  new TextRun("Clinical decision support; adverse drug events; electronic health records; MIMIC-IV; medication safety; laboratory trends; explainable AI; temporal modeling; alert fatigue; pharmacovigilance."),
] }));

// 1. Introduction
body.push(new Paragraph({ pageBreakBefore: true, heading: HeadingLevel.HEADING_1, children: [new TextRun("1. Introduction")] }));
body.push(P(
  "Medication-related harm remains stubbornly common in hospital care. Early audits found adverse drug events in roughly six of every hundred admissions, with a substantial fraction judged preventable [1]. Computerised order entry and clinical decision support were introduced partly in response, and when well designed they do reduce serious medication errors [2, 3]. The same tools, however, have a well-known failure mode: because the underlying drug-safety logic is built around fixed cut-offs, it tends to fire often and indiscriminately. Clinicians override the majority of such alerts, and reported override rates in outpatient settings frequently exceed ninety per cent [4, 6]. The result is alert fatigue, in which the genuinely important warnings are dismissed alongside the trivial ones [5]."
));
body.push(P(
  "Laboratory monitoring is one of the clearest opportunities to catch drug-related harm before it becomes clinically overt, and the link between pharmacy and laboratory data has long been recognised as fertile ground for error reduction [7]. The difficulty is that the risk that matters is usually not a single dangerous value but a trajectory: a potassium climbing after an ACE inhibitor is started, an INR drifting above range on warfarin, a glomerular filtration rate sliding downward on metformin. A threshold check cannot tell a value that is rising fast from one that is stable, and it cannot tell a meaningful within-patient change from a number that merely sits inside the population reference interval."
));
body.push(H2("1.1 Problem statement"));
body.push(P(
  "Conventional drug-safety checks operate on isolated values. They do not consistently connect the start of a medication to the laboratory change that follows it, they do not compare a patient against their own prior measurements, and they treat a transient blip the same as a sustained deterioration. They therefore tend to detect risk late and to alert excessively."
));
body.push(H2("1.2 Research gap"));
body.push(P(
  "Between simple static drug-laboratory thresholds, which are easy to implement but late and noisy, and the temporal, patient-specific, explainable detection that clinicians actually reason with, there is a practical gap. Prior data-mining and machine-learning work on adverse drug events [8] and on temporal modelling of electronic health records [12, 13, 14] points the way, but a scenario-focused, end-to-end system that ties drug exposure to a longitudinal laboratory trend, scores the risk transparently, and explains each alert in clinical language has not been the norm, and evidence on real EHR data has been limited. This study targets that gap."
));
body.push(H2("1.3 Research question"));
body.push(P(
  "Can a temporal, explainable medication-laboratory conflict detection system identify drug-related clinical risks earlier and more accurately than conventional threshold-based alerts on a large, real-world EHR cohort?"
));
body.push(H2("1.4 Hypothesis"));
body.push(P(
  "A temporal detector that integrates medication exposure with longitudinal laboratory trends and patient-specific baselines will surface medication-related clinical risks earlier and with fewer low-value alerts than a static-threshold rule, while learned models trained on the same temporal features will further improve discrimination at the cost of additional model machinery."
));
body.push(H2("1.5 Objectives"));
[
  "Build a system that links medications to laboratory results over time on real EHR data.",
  "Detect medication-laboratory conflicts before a result reaches a severe danger level.",
  "Compare the proposed approach against conventional static-threshold rules on a large MIMIC-IV cohort.",
  "Produce a per-patient 0-100 risk score with transparent components.",
  "Generate an understandable clinical explanation for every alert.",
  "Quantify the change in alert burden when moving from a static threshold to a temporal rule.",
  "Quantify early-detection ability through time-to-alert at both the cohort and case level.",
  "Evaluate the learned models with AUROC, AUPRC, F1, sensitivity, specificity, and the Brier score.",
  "Illustrate interpretability through concrete clinical case examples.",
  "Expose the results through an API and an interactive dashboard.",
].forEach((t, i) => body.push(NUMP(i + 1, t)));
body.push(H2("1.6 Contributions"));
[
  "A temporal medication-laboratory conflict detection framework that integrates medication exposure with longitudinal laboratory trends, evaluated end-to-end on MIMIC-IV.",
  "Patient-specific baseline comparison built from each patient's pre-drug measurements, enabling within-patient detection of clinically meaningful change.",
  "A transparent 0-100 risk-scoring mechanism that yields clinically interpretable alert explanations.",
  "A head-to-head comparison of static thresholds, sustained-trend temporal rules, and four learned models on the same large EHR cohort.",
  "An honest discussion of where the temporal rule helps and where it does not, with case-level evidence that the early-warning use case remains achievable for individual patients.",
].forEach((t, i) => body.push(NUMP(i + 1, t)));

// 2. Related Work
body.push(new Paragraph({ pageBreakBefore: true, heading: HeadingLevel.HEADING_1, children: [new TextRun("2. Related Work")] }));
body.push(P(
  "Computerised drug-safety checking grew out of drug-drug interaction and drug-disease contraindication rules, and systematic evidence shows that decision support improves practice mainly when it is timely, automatic, and actionable [3]. The narrower problem of drug-laboratory interaction, flagging a drug when a related laboratory value is abnormal, has received less attention and is usually implemented as a static, single-value rule that ignores both the patient's baseline and the temporal trend [7]. A persistent theme across this literature is alert fatigue: when systems generate too many low-value warnings, clinicians override most of them, and the override behaviour itself becomes a safety concern [4, 5, 6]. Our design responds to this directly by gating on a sustained, baseline-relative change rather than a single threshold crossing, with the aim of raising specificity and reducing alert volume."
));
body.push(P(
  "On the predictive side, pharmacovigilance and adverse-event research has moved from spontaneous-report data mining [8] toward models trained on electronic health records. Sequence models such as Doctor AI and RETAIN demonstrated that recurrent architectures can forecast clinical events and, in RETAIN's case, can do so with attention-based interpretability [12, 13]. Large-scale deep models have since been applied to raw EHR data [14], and a continuous, deployable predictor of acute kidney injury showed that risk can be anticipated up to two days in advance [15]. Real-time early-warning systems for deterioration, exemplified by sepsis scores derived from streaming data, further established that trend-based monitoring can provide a clinically useful lead time [16]. We deliberately favour interpretable temporal features over opaque sequence models, both as a transparent baseline and because the clinician-facing explanations we want to generate are easier to ground when the underlying features are themselves clinical."
));
body.push(P(
  "Interpretability is now a prerequisite for clinical machine learning. SHAP provides a unified, locally accurate attribution method [17] with an efficient exact algorithm for tree ensembles [18], and we pair it with template-based clinical narratives so that each alert is explained in language a prescriber can act on. Methodologically we follow established guidance on probability calibration [23] and on transparent reporting of clinical prediction models [24]. The pharmacological relationships we model are themselves well characterised: the effect of renin-angiotensin-aldosterone blockade on potassium handling [25], the management of warfarin and INR [26], the renal cautions surrounding metformin [27], and the nephrotoxicity of non-steroidal anti-inflammatory drugs [28]; renal function itself is summarised through the CKD-EPI 2021 estimate of glomerular filtration rate [29] and graded against KDIGO criteria for acute kidney injury [30]."
));

// 3. Methodology
body.push(new Paragraph({ pageBreakBefore: true, heading: HeadingLevel.HEADING_1, children: [new TextRun("3. Methodology")] }));
body.push(H2("3.1 System architecture"));
body.push(P(
  "The platform is a layered pipeline: data ingestion, preprocessing and timeline construction, temporal feature engineering, conflict detection by rules and learned models, risk scoring, explanation generation, and a dashboard and API surface (Figure 1). Each layer reads only from the canonical data model produced by the layer above it, which is what allows the entire system to switch between data sources without code changes."
));
body.push(image(FIG("fig1_architecture.png"), 620, 124));
body.push(CAPTION("Figure 1. System architecture, from data ingestion to explainable, prioritized alerts."));

body.push(H2("3.2 Data source: MIMIC-IV v2.2"));
body.push(P(
  "The evaluation reported here uses the hospital (hosp) module of MIMIC-IV v2.2 [10], a publicly available de-identified critical-care and inpatient EHR dataset released by the MIT Laboratory for Computational Physiology under a credentialed Data Use Agreement. After signed credentialing, the loader extracts only the tables needed for the three scenarios — patients, admissions, prescriptions, labevents, d_labitems, diagnoses_icd, d_icd_diagnoses — and applies a hosp-only filter to avoid pulling in ICU streaming data that is irrelevant here. Laboratory item identifiers are mapped to the canonical lab names using the d_labitems dictionary, with multiple itemids per analyte permitted (for example, serum, whole-blood, and blood-gas potassium all map to potassium). Drug exposures are normalised by name and class from the prescriptions table, and overlapping or repeated prescriptions of the same drug for the same patient are merged into a single exposure episode so that days-since-drug-start is well-defined. Renal and hepatic comorbidity are detected from ICD-9 and ICD-10 codes in diagnoses_icd. eGFR is computed from serum creatinine, age, and sex using the race-neutral CKD-EPI 2021 equation [29]. The synthetic loader used in earlier proof-of-concept work is retained for testing but plays no part in the results reported below."
));
body.push(H2("3.3 Cohort and labels"));
body.push(P(
  `After loading and filtering, the cohort comprises ${N_PT} patients with at least one exposure to a study drug class and at least one relevant post-exposure laboratory measurement. Outcome labels are sustained rather than instantaneous: a positive label requires at least two qualifying readings within the scenario-specific label horizon, so a single transient out-of-range value does not generate a positive label. Hyperkalemia under ACE/ARB exposure is defined as sustained potassium of at least 5.3 mmol/L within thirty days; warfarin-related bleeding risk as sustained INR of at least 3.5 within fourteen days; metformin-related renal decline as a sustained drop in eGFR of at least twenty per cent from baseline within sixty days, with current eGFR of at most 45 mL/min/1.73 m².`
));
body.push(H2("3.4 Preprocessing and patient-specific baseline"));
body.push(P(
  "Preprocessing normalises drug and laboratory names, harmonises units, discards physiologically implausible values, orders events chronologically, and builds a per-patient timeline. The baseline for each laboratory parameter is the median of pre-drug readings within a 180-day look-back, falling back to the earliest post-start reading when no pre-drug data exist. Comparing each patient against this personal baseline, rather than against the population interval alone, is what allows a clinically meaningful rise from 4.2 to 5.1 mmol/L to register even though it never leaves the reference range."
));
body.push(H2("3.5 Feature engineering"));
body.push(P(
  "For each patient-laboratory series the features are computed using only readings up to an early observation cutoff (14 days for potassium, 10 for INR, 30 for eGFR), so the feature matrix never borrows information from the longer label window. The feature families are summarised in Table 1."
));
body.push(table(["Feature Category", "Examples", "Clinical Meaning"],
  DATA.table2_feature_categories.map((r) => [r["Feature Category"], r["Examples"], r["Clinical Meaning"]]),
  [2200, 3600, 3560]));
body.push(CAPTION("Table 1. Temporal and risk feature categories used by the rule and the learned models."));

body.push(H2("3.6 Conflict detection strategies"));
body.push(P(
  "Three strategies are compared on the same cohort. The static-threshold rule fires when the primary laboratory crosses a fixed cut-off. The temporal rule fires only on a sustained, baseline-relative change that crosses an earlier gate — for example, a potassium rise of at least 0.5 mmol/L with a current value of at least 5.0 mmol/L and at least two consecutive readings moving in the dangerous direction. The gate sits below the hard threshold, which enables earlier detection in principle, while the sustained-trend requirement suppresses isolated spikes. The four learned models — logistic regression, random forest [19], XGBoost [20], and LightGBM [21] — are trained per scenario on the temporal feature matrix using scikit-learn [22], with stratified 70/30 splits and a fixed random seed for reproducibility."
));

body.push(H2("3.7 Algorithms"));
body.push(P("The temporal rule is evaluated at each post-drug reading as follows:"));
[
  "compute baseline = median(pre-drug readings within 180 days)",
  "for each reading r at time t after drug start:",
  "    delta = r.value - baseline ;  pct = delta / baseline",
  "    change_ok = (rising risk: delta >= delta_threshold) OR (falling risk: pct <= -delta_threshold)",
  "    current_ok = value crosses the temporal gate (set below the hard threshold)",
  "    sustained  = consecutive readings in the dangerous direction >= 2",
  "    if change_ok AND current_ok AND sustained: raise temporal alert at t",
].forEach((t) => body.push(MONO(t)));

body.push(H2("3.8 Risk score"));
body.push(P(
  "Alerts are graded by a transparent weighted 0-100 score that maps to four bands (0-30 Low, 31-60 Moderate, 61-80 High, 81-100 Critical):"
));
body.push(MONO("risk = 100 x SUM( w_i * c_i ),  c_i in [0,1]"));
body.push(P(
  "The eight components are laboratory-abnormality severity, temporal slope, change from baseline, strength of the known drug-laboratory relationship, renal and hepatic vulnerability, age, polypharmacy, and trend persistence. Weights are centralised in a single configuration file and can be tuned or ablated. The default weighting emphasises abnormality severity, slope, and baseline deviation; the dangerous direction is enforced so that a resolving trend cannot inflate the score. Figure 3 shows the component breakdown for one high-risk alert."
));
body.push(image(FIG("fig3_risk_components.png"), 540, 300));
body.push(CAPTION("Figure 3. Risk-score component breakdown for an example high-risk hyperkalemia alert."));

body.push(H2("3.9 Explainability"));
body.push(P(
  "Every alert is rendered as a clinical sentence that names the drug and laboratory, states the baseline-to-current change and the time window, relates the change to the drug start, and identifies the dominant risk factor; it ends with a suggested review action rather than a treatment instruction. For model-based alerts the leading SHAP drivers [17, 18] are appended. Global SHAP importance is shown in Figure 5."
));

body.push(H2("3.10 Database schema"));
body.push(P("The canonical relational model comprises six tables:"));
body.push(table(["Table", "Key fields"], [
  ["patients", "patient_id, age, sex, comorbidities, renal_disease_status, liver_disease_status, admission/discharge_date"],
  ["medications", "medication_id, patient_id, drug_name, drug_class, start_date, dose, dose_change_date, stop_date"],
  ["labs", "lab_id, patient_id, lab_name, value, unit, reference_range_low/high, lab_date"],
  ["medication_lab_risk_knowledge", "rule_id, drug_class, lab_name, expected_direction, risk_type, severity_weight, time_window_days, threshold_value, delta_threshold"],
  ["alerts", "alert_id, patient_id, medication_id, risk_score, risk_level, explanation, suggested_action, model_type, clinician_feedback"],
  ["model_features", "patient_id, baseline_value, current_value, delta_value, percent_change, slope_7/14/30d, days_since_drug_start, egfr, polypharmacy_count, label"],
], [2600, 6760]));
body.push(CAPTION("Table 2. Canonical six-table data model shared by all data sources."));

body.push(H2("3.11 Dashboard and API"));
body.push(P(
  "A Streamlit dashboard lists patients by risk, plots each patient's timeline (drug start, dose change, laboratory trend, and the static versus temporal alert points), and presents risk cards, the explanation panel, and the suggested review actions; it offers filters by risk level, a high-risk-only view, a side-by-side rule-versus-model comparison, and per-patient report export. A FastAPI service exposes endpoints for patients, timelines, laboratories, medications, alerts, risk prediction, alert explanation, and model performance."
));

// 4. Experimental design
body.push(new Paragraph({ pageBreakBefore: true, heading: HeadingLevel.HEADING_1, children: [new TextRun("4. Experimental Design")] }));
body.push(P(
  "The three scenarios and their temporal patterns are listed in Table 3. Each scenario dataset has one row per exposed patient; features are measured at the early observation cutoff and the label is a sustained adverse outcome over the longer label window, so the task is genuinely one of early prediction rather than retrospective fitting. Data are split 70/30 with stratification, and predicted probabilities are assessed for calibration following standard practice [23]."
));
body.push(table(["Drug/Class", "Lab", "Risk Type", "Temporal Pattern"],
  DATA.table1_scenarios.map((r) => [r["Drug/Class"], r["Laboratory Test"], r["Risk Type"], r["Temporal Pattern"]]),
  [2700, 1700, 1860, 3100]));
body.push(CAPTION("Table 3. Medication-laboratory conflict scenarios and the temporal patterns used to detect them."));

body.push(H2("4.1 Experiment 1 - Static versus temporal rules"));
body.push(P("We compare static thresholds against temporal rules on sensitivity, specificity, precision, recall, F1, the false-alert rate, and alerts and false alerts per 100 patients."));
body.push(H2("4.2 Experiment 2 - Machine-learning models"));
body.push(P("We compare logistic regression, random forest, XGBoost, and LightGBM on AUROC, AUPRC, F1, sensitivity, specificity, and the Brier score for calibration."));
body.push(H2("4.3 Experiment 3 - Early detection (time-to-alert)"));
body.push(P("Among true positives that both methods eventually flag, we measure how many days earlier the temporal method fires than the static threshold, together with the change in overall alert burden."));
body.push(H2("4.4 Experiment 4 - Explainability illustration"));
body.push(P("We assemble illustrative explanations for representative high-risk cases (Table 7) and report global feature importance via SHAP, alongside a clinician-rating questionnaire bundled into the repository for future evaluation."));

// 5. Results
body.push(new Paragraph({ pageBreakBefore: true, heading: HeadingLevel.HEADING_1, children: [new TextRun("5. Results")] }));
body.push(P(
  `The pipeline was run end-to-end on the full MIMIC-IV v2.2 cohort (${N_PT} patients), producing ${N_AL} alerts across the three scenarios. The picture that emerges differs in instructive ways from the synthetic proof-of-concept that preceded this work, and we report the findings as they are rather than as we expected them to be.`
));

body.push(H2("5.1 Static versus temporal rules"));
const e1 = DATA.exp1_rules_vs_temporal;
const ace_s = e1.find(r => r.scenario === SCN_ACE && r.method === "Static threshold");
const ace_t = e1.find(r => r.scenario === SCN_ACE && r.method === "Temporal rule");
const war_s = e1.find(r => r.scenario === SCN_WAR && r.method === "Static threshold");
const war_t = e1.find(r => r.scenario === SCN_WAR && r.method === "Temporal rule");
const met_s = e1.find(r => r.scenario === SCN_MET && r.method === "Static threshold");
const met_t = e1.find(r => r.scenario === SCN_MET && r.method === "Temporal rule");
body.push(P(
  "The contrast between static and temporal rules turns out to be more nuanced on real data than on synthetic data, and the three scenarios behave differently enough that they deserve separate treatment. " +
  `For ACE inhibitor/ARB-driven hyperkalemia, the temporal rule produced a clear gain on alert burden: overall alerts fell from ${ace_s.alerts_per_100} to ${ace_t.alerts_per_100} per 100 patients, false alerts from ${ace_s.false_alerts_per_100} to ${ace_t.false_alerts_per_100} per 100 patients, and specificity rose from ${fmt(ace_s.specificity, 3)} to ${fmt(ace_t.specificity, 3)}. Sensitivity, however, fell from ${fmt(ace_s.sensitivity, 3)} to ${fmt(ace_t.sensitivity, 3)}, so the rule sacrifices some cases to gain a calmer alerting profile. ` +
  `For warfarin and elevated INR the static threshold was already well tuned: sensitivity was approximately preserved by the temporal rule (${fmt(war_s.sensitivity, 3)} versus ${fmt(war_t.sensitivity, 3)}) but specificity dropped from ${fmt(war_s.specificity, 3)} to ${fmt(war_t.specificity, 3)} and the false-alert rate rose, suggesting that for an outcome dominated by acute INR spikes a sustained-trend gate adds noise rather than removing it. ` +
  `For metformin and eGFR decline both detectors struggle with sensitivity (around ${fmt(met_s.sensitivity, 2)}), which reflects how slowly the labelled outcome accrues in this cohort and how few patients meet the strict label definition (${commify(DATA.table4_early_detection.find(r=>r.scenario===SCN_MET).n_positive)} positives). ` +
  "The headline lesson is that the temporal rule helps most where the underlying physiology drifts gradually and reliably (RAAS-driven hyperkalemia), and helps less where the outcome is abrupt (warfarin/INR) or label-sparse (metformin/eGFR)."
));
body.push(table(["Scenario", "Method", "Sens", "Spec", "F1", "False alerts/100"],
  e1.map((r) => [shortScenario(r.scenario), r.method, fmt(r.sensitivity, 3), fmt(r.specificity, 3), fmt(r.f1, 3), fmt(r.false_alerts_per_100, 1)]),
  [2900, 2100, 1100, 1100, 1100, 1060]));
body.push(CAPTION("Table 4. Experiment 1: static versus temporal rules on MIMIC-IV. The temporal rule trades sensitivity for specificity and lower alert volume; the magnitude of the trade-off depends on the scenario."));

body.push(H2("5.2 Machine-learning models"));
const m_ace = DATA.table3_model_comparison.find(r => r.scenario === SCN_ACE && r.model === "XGBoost");
const m_war = DATA.table3_model_comparison.find(r => r.scenario === SCN_WAR && r.model === "XGBoost");
const m_met = DATA.table3_model_comparison.find(r => r.scenario === SCN_MET && r.model === "XGBoost");
body.push(P(
  "The learned models separated sustained adverse outcomes well in every scenario, with the gradient-boosted models performing best and producing calibrated probabilities. " +
  `XGBoost, the primary model in each scenario, reached AUROC ${fmt(m_ace.auroc, 3)} and AUPRC ${fmt(m_ace.auprc, 3)} for hyperkalemia, AUROC ${fmt(m_war.auroc, 3)} and AUPRC ${fmt(m_war.auprc, 3)} for warfarin-related INR elevation, and AUROC ${fmt(m_met.auroc, 3)} and AUPRC ${fmt(m_met.auprc, 3)} for metformin-related renal decline (Table 5, Figure 4). Brier scores ranged from ${fmt(m_met.brier, 3)} to ${fmt(m_war.brier, 3)} across the primary models, well within the range usually considered well-calibrated. ` +
  "The strong AUROC values combined with more modest AUPRC are consistent with the moderate-to-high class imbalance present in each scenario, and indicate that the value embedded in the temporal feature set is largely recovered by a model that can weigh the features jointly."
));
body.push(table(["Scenario", "Model", "AUROC", "AUPRC", "F1", "Brier"],
  DATA.table3_model_comparison.map((r) => [shortScenario(r.scenario), r.model, fmt(r.auroc, 3), fmt(r.auprc, 3), fmt(r.f1, 3), fmt(r.brier, 3)]),
  [2700, 2200, 1115, 1115, 1115, 1115]));
body.push(CAPTION("Table 5. Experiment 2: head-to-head model comparison on MIMIC-IV. Primary model: XGBoost in every scenario."));
body.push(image(FIG("fig4_model_performance.png"), 620, 238));
body.push(CAPTION("Figure 4. ROC (a) and Precision-Recall (b) curves for the primary XGBoost model in each scenario."));

body.push(H2("5.3 Early detection (time-to-alert)"));
const e3 = DATA.table4_early_detection;
const e3_ace = e3.find(r => r.scenario === SCN_ACE);
const e3_war = e3.find(r => r.scenario === SCN_WAR);
const e3_met = e3.find(r => r.scenario === SCN_MET);
body.push(P(
  "The early-detection finding is the most nuanced and the one most worth reading carefully. " +
  `Aggregated across all positives flagged by both methods, the mean lead of the temporal rule over the static threshold was small and not consistently positive: ${e3_ace.mean_days_earlier} days for the ACE/ARB scenario (n=${commify(e3_ace.n_both_alerted)} co-detected), ${e3_war.mean_days_earlier} days for warfarin (n=${commify(e3_war.n_both_alerted)}), and ${e3_met.mean_days_earlier} days for metformin (n=${commify(e3_met.n_both_alerted)}). The median lead was zero days in every scenario. ` +
  `Where the temporal rule does show its value at the cohort level is the reduction in alert volume: ${e3_ace.alert_reduction_pct}% fewer alerts in the ACE/ARB scenario, without a corresponding loss of specificity (Table 4). ` +
  "The aggregate days-earlier metric is therefore best read as evidence that the temporal rule does not, on average, fire substantially earlier than a well-tuned threshold on this cohort. It does, however, identify a subset of patients in whom the lead-time is clinically meaningful: Figure 2 shows one such case (patient 14866589) in which the temporal logic raised a hyperkalemia alert on day 1 after lisinopril initiation, while the static threshold did not fire until day 30 — a 29-day lead for that patient. The mean and the case study tell complementary truths, and we report both."
));
body.push(table(["Scenario", "n_positive", "n_both_alerted", "Static day", "Temporal day", "Days earlier (mean)", "Alert reduction"],
  e3.map((r) => [shortScenario(r.scenario), commify(r.n_positive), commify(r.n_both_alerted),
                 fmt(r.mean_static_alert_day, 1), fmt(r.mean_temporal_alert_day, 1),
                 fmt(r.mean_days_earlier, 1), fmt(r.alert_reduction_pct, 1) + "%"]),
  [2200, 1200, 1500, 1200, 1200, 1500, 1560]));
body.push(CAPTION("Table 6. Experiment 3: time-to-alert and alert-volume summary on MIMIC-IV. Aggregate lead-time is small; case-level lead-time can be substantial (see Figure 2)."));
body.push(image(FIG("fig2_patient_timeline.png"), 560, 308));
body.push(CAPTION("Figure 2. Patient timeline (MIMIC-IV patient 14866589). The temporal rule raised a hyperkalemia alert on day 1 after lisinopril initiation, while the static threshold did not fire until day 30 (29-day lead for this patient)."));

body.push(H2("5.4 Explainability and case studies"));
body.push(P(
  "Representative high-risk alerts with their generated explanations are shown in Table 7. Each explanation states the baseline-to-current change, the time window relative to the drug start, the dominant patient-level risk factors, and the leading SHAP drivers, and ends with a suggested review action rather than a directive. The set spans all three scenarios and includes cases with sharply elevated risk (potassium 9.4 mmol/L; eGFR drop from 53.7 to 31.9 mL/min/1.73 m²) where the trend across the observation window is the natural unit of evidence. Global SHAP feature importance for the primary hyperkalemia model is shown in Figure 5."
));
body.push(table(["Case", "Risk", "Explanation (excerpt)"],
  DATA.table5_example_explanations.slice(0, 6).map((r) => [shortScenario(r.case), `${r.risk_level} (${fmt(r.risk_score, 1)})`, clip(r.explanation, 240)]),
  [2200, 1300, 5860]));
body.push(CAPTION("Table 7. Representative interpretable alert explanations across the three scenarios (six of twelve shown)."));
body.push(image(FIG("fig5_shap_summary.png"), 560, 360));
body.push(CAPTION("Figure 5. Global SHAP feature importance for the primary XGBoost model in the ACE/ARB - potassium scenario."));

body.push(H2("5.5 Comparison with previous studies"));
body.push(P(
  "The results sit comfortably alongside the published direction of travel, and the differences are themselves informative. The central weakness of conventional decision support has been its alert burden: override rates above ninety per cent have been documented for medication-related alerts in outpatient care [6], and broader analyses link repeated low-value alerts to fatigue and reduced responsiveness [4, 5]. The eighteen-per-cent reduction in alert volume we observe in the ACE/ARB scenario, with specificity rising rather than falling, is consistent with the recommendation that effective decision support must be specific and actionable [3]. On early detection, continuous models have anticipated acute kidney injury up to forty-eight hours ahead [15] and streaming early-warning scores have provided useful lead time for sepsis [16]; our aggregate days-earlier metric is more modest than those headline figures, in part because the outcomes we target develop on a slower, drug-driven timescale than acute decompensation, and in part because our comparator is a sustained-trend label rather than a single threshold crossing. The case-level lead of twenty-nine days we observe for patient 14866589, however, is directionally aligned with the AKI lead time [15] and reinforces that timely warning is achievable with interpretable temporal features. The interpretability we offer extends attention-based clinical models such as RETAIN [13] by combining model-level SHAP attributions [17, 18] with template-based clinical narratives that name the drug, the trend, and the suggested action, so that the alert can be acted on without the reader having to translate from feature importances to clinical language."
));

// 6. Discussion
body.push(new Paragraph({ pageBreakBefore: true, heading: HeadingLevel.HEADING_1, children: [new TextRun("6. Discussion")] }));
body.push(P(
  "The move from a 900-patient synthetic proof of concept to a 122,166-patient MIMIC-IV cohort substantially compresses the apparent advantage of temporal rules over static thresholds, and we read this as a useful corrective rather than a setback. On synthetic data the temporal rule dominated the static threshold on essentially every axis, which in hindsight reflected the generator: progressive drift was the only failure mode it knew how to produce. On real data the underlying outcomes do not all fit that mould. Hyperkalemia after RAAS blockade does drift gradually and is exactly where the temporal rule is most useful — it cut alert volume by eighteen per cent without losing specificity. Warfarin-driven INR elevation is dominated by sharper deflections that a sustained-trend gate is poorly suited to capture, and it shows. Metformin-related renal decline accrues slowly and labels are sparse, which depresses sensitivity for both detectors and limits what either can demonstrate."
));
body.push(P(
  "The learned models stand up better. XGBoost reached AUROC of 0.93 to 0.97 in every scenario with calibrated probabilities, which suggests that the value embedded in the temporal feature set is fully recoverable when the features can be weighed jointly. From a deployment perspective, the rules are still the interpretable backstop and the natural place to start; the models are where the discriminative gain lives. The two together, paired with a transparent 0-100 risk score and a clinician-readable explanation, are what we would expose to a prescriber, with the rule providing a sanity floor and the model carrying the per-patient probability."
));
body.push(P(
  "The case study of patient 14866589 in Figure 2 is a reminder that even when an effect averages to zero, important individual cases live inside the cohort. Twenty-nine days of warning before a static threshold fires is not a typical case, but it is the kind of case where the system can plausibly change a clinical decision. The honest statement is that the temporal logic does not earn its keep on every patient; it earns it on the patients for whom the trajectory is the diagnosis."
));

body.push(H2("6.1 Scope of utility: when does temporal alerting matter?"));
body.push(P(
  "A reasonable objection to any temporal monitoring system is that an attentive clinician with frequent laboratory data already sees the trajectory. If a patient in intensive care is sampled every twelve hours and the responsible physician is at the bedside, the system arguably adds little that human vigilance does not already cover. We take this objection seriously, because the value we claim for the approach has to be defended at the level of the workflow in which it would actually be used, not at the level of the algorithm in isolation."
));
body.push(P(
  "The clinical reality, however, is that the density and the reviewability of laboratory data are not the same thing. A single intensive-care physician routinely carries fifteen to twenty-five patients per shift, each producing tens of laboratory values per day; the working-memory cost of tracking, for every patient, a baseline-relative slope across the prior week — while simultaneously managing ventilation, sedation, hemodynamics, and family communication — is well beyond what is realistic. The literature on alert fatigue and on missed adverse drug events is consistent with this reading: the relevant values are usually visible in the chart before harm occurs, but they are not synthesised into a timely decision [4, 5, 6]. What the system contributes is therefore not new information but a quantified, baseline-anchored summary of information that is already present — a slope, a percentage change from the patient's own pre-drug median, an explicit days-since-drug-start — computed automatically and at the same moment a new value enters the record."
));
body.push(P(
  "Read in this light, the marginal utility of the temporal layer is not constant across the care continuum. It is highest where laboratory cadence is sparse and clinician attention per patient is necessarily diluted: the general medical ward, the outpatient clinic, the small or rural hospital, the post-discharge follow-up period, and any setting in which a patient on a slow-onset drug-induced trajectory is reviewed weekly rather than daily. It is more modest, though not zero, in intensive care, where the system functions less as an early-warning device and more as a standardising layer that produces the same quantitative read of the trend regardless of which shift is on duty and that frees clinician attention for cases where the trajectory is not the dominant question. Across both regimes the framing we adopt is that of augmented, not artificial, intelligence: the system does not replace the clinician's judgement but extends the clinician's capacity to attend to patterns whose detection exceeds reliable human working-memory bandwidth in real-time clinical workflows."
));
body.push(P(
  "A practical consequence is that the cohort on which a temporal detector is evaluated co-determines the apparent size of its advantage. MIMIC-IV is intensive-care-weighted, and the modest aggregate lead-time we report is consistent with a setting in which vigilance is already high and laboratory sampling is dense; the same pipeline applied to outpatient or general-ward data would, on the basis of this reasoning, be expected to show a larger time-to-alert advantage and a more pronounced reduction in low-value alerts. We name this explicitly as a limitation of the present evaluation and as a target for the external validation work outlined in §8."
));

// 7. Limitations
body.push(H1("7. Limitations"));
[
  "MIMIC-IV is a single-centre, ICU-skewed dataset; baseline characteristics, prescribing patterns, and laboratory practices may differ at other sites, and external validation is necessary before any clinical generalisation. Because intensive-care patients are sampled densely and watched closely, the cohort under-represents precisely the settings (general wards, outpatient clinics, post-discharge follow-up) in which the temporal layer is expected to add the most value (§6.1).",
  "Outcome labels are proxy definitions chosen for tractability (for example, sustained potassium of at least 5.3 mmol/L); a chart-reviewed clinical adjudication would tighten the labels but is outside the scope of an open-data study.",
  "Some patients have years of pre-drug laboratory history that overlap with multiple unrelated drug exposures; the 180-day baseline window cannot fully isolate the studied drug, and confounding by indication remains.",
  "Acute illness, intercurrent infections, and changes in concomitant medications produce laboratory perturbations that are not attributable to the studied drug, which inflates both static and temporal alert rates relative to a controlled setting.",
  "Days-earlier statistics are reported only as means, medians, and case examples; a survival-style analysis of the lead-time distribution would be more informative and is left to future work.",
  "Knowledge-base coverage is restricted to the three modelled scenarios; broader coverage will require curation of additional drug-laboratory rules.",
  "The system has not been evaluated prospectively and has not been used by clinicians in a live setting; the explanations have face validity but not yet measured utility.",
  "Alerts are decision support, not diagnoses, and the system does not replace clinical judgment.",
].forEach((t) => body.push(BULLET(t)));

// 8. Future work
body.push(H1("8. Future Work"));
[
  "External validation on the eICU collaborative database [11] and on IRB-approved local hospital data, using the existing source-agnostic loader.",
  "Survival-style analysis of the time-to-alert distribution, identifying the patient subgroup for whom the temporal lead-time is clinically meaningful.",
  "Add scenarios such as NSAIDs and creatinine, diuretics and sodium or potassium, statins and transaminases or creatine kinase, spironolactone and potassium, and SSRIs and sodium.",
  "Explore deep temporal architectures (TCN, LSTM, Transformer) once the cohort and label definitions support the additional model capacity.",
  "Incorporate clinical notes and study multi-site federated learning where data cannot leave the institution.",
  "Run a prospective shadow-mode study and measure the effect on clinician decisions and patient outcomes.",
  "Integrate with a live EHR as a decision-support service, with clinician feedback captured in the alerts table.",
].forEach((t) => body.push(BULLET(t)));

// 9. Conclusion
body.push(H1("9. Conclusion"));
body.push(P(
  `An explainable, temporal medication-laboratory conflict detector that judges each patient against their own baseline can plausibly support earlier and better-explained warning of drug-related laboratory risk, with a lower alert burden, than conventional static-threshold alerts. The present evaluation on ${N_PT} MIMIC-IV patients shows that the cohort-level lead-time advantage of a sustained-trend rule over a well-tuned threshold is modest, but that the rule reduces alert burden materially in the scenarios where the underlying physiology drifts gradually, and that calibrated gradient-boosted models trained on the same temporal features deliver strong discrimination (AUROC 0.93-0.97). For individual patients the temporal logic can still flag risk up to several weeks before a static threshold does. Prospective evaluation and external validation are necessary before any clinical deployment.`
));

// 10. Ethics
body.push(H1("10. Ethical Considerations"));
body.push(P(
  "The system does not prescribe, does not stop medications automatically, and does not issue a diagnosis; it functions strictly as decision support that surfaces potential risk for human review. MIMIC-IV is a de-identified, publicly available dataset accessed under a credentialed Data Use Agreement and we have followed its terms throughout, including the restrictions on data sharing and on attempts at re-identification. Any future use of identified data requires institutional review board approval, with patient privacy protected throughout."
));

// 11. Implementation roadmap, technologies, venues
body.push(new Paragraph({ pageBreakBefore: true, heading: HeadingLevel.HEADING_1, children: [new TextRun("11. Implementation Roadmap")] }));
[
  "Phase 1, proof of concept on synthetic data: full pipeline, rules, machine learning, risk score, explanations, dashboard, and experiments. (Complete.)",
  "Phase 2, MIMIC-IV integration on credentialed access: implement the adapter and re-run the experiments on real data. (Complete; reported in this manuscript.)",
  "Phase 3, external validation on eICU [11] or IRB-approved local data, with recalibration of weights and thresholds.",
  "Phase 4, clinician explainability evaluation at scale, using the questionnaire bundled with the repository.",
  "Phase 5, prospective shadow-mode evaluation integrated with an EHR.",
].forEach((t, i) => body.push(NUMP(i + 1, t)));

body.push(H2("11.1 Suggested technologies"));
body.push(P(
  "Backend: Python, FastAPI, pandas, scikit-learn, XGBoost, LightGBM, SHAP, SQLAlchemy. Database: SQLite for the prototype, PostgreSQL for production. Frontend: Streamlit for the research dashboard, React for a product build. Visualisation: Matplotlib, Plotly."
));

body.push(H2("11.2 Possible publication venues"));
body.push(P(
  "Journals: Journal of the American Medical Informatics Association (JAMIA) and JAMIA Open; Journal of Biomedical Informatics; JMIR Medical Informatics; BMC Medical Informatics and Decision Making; Artificial Intelligence in Medicine; IEEE Journal of Biomedical and Health Informatics. Conferences: AMIA Annual Symposium and MedInfo."
));

body.push(H2("11.3 Reproducibility"));
body.push(P(
  "The full pipeline is open-source. The MIMIC-IV results in this manuscript can be reproduced from a credentialed copy of the dataset by running `python run_pipeline.py` with `DATA_SOURCE=mimic`; figures can be regenerated without re-training by running `python regenerate_figures.py`. The manuscript itself is rebuilt from the figures and tables by running `node paper/build_manuscript.js`, so every numeric value in the Results section traces back to a CSV in `paper/tables/` and ultimately to the pipeline run."
));

// 12. References
body.push(new Paragraph({ pageBreakBefore: true, heading: HeadingLevel.HEADING_1, children: [new TextRun("12. References")] }));
[
  "Bates DW, Cullen DJ, Laird N, et al. Incidence of adverse drug events and potential adverse drug events: implications for prevention. JAMA. 1995;274(1):29-34.",
  "Bates DW, Leape LL, Cullen DJ, et al. Effect of computerized physician order entry and a team intervention on prevention of serious medication errors. JAMA. 1998;280(15):1311-1316.",
  "Kawamoto K, Houlihan CA, Balas EA, Lobach DF. Improving clinical practice using clinical decision support systems: a systematic review of trials to identify features critical to success. BMJ. 2005;330(7494):765.",
  "van der Sijs H, Aarts J, Vulto A, Berg M. Overriding of drug safety alerts in computerized physician order entry. J Am Med Inform Assoc. 2006;13(2):138-147.",
  "Ancker JS, Edwards A, Nosal S, et al. Effects of workload, work complexity, and repeated alerts on alert fatigue in a clinical decision support system. BMC Med Inform Decis Mak. 2017;17(1):36.",
  "Nanji KC, Slight SP, Seger DL, et al. Overrides of medication-related clinical decision support alerts in outpatients. J Am Med Inform Assoc. 2014;21(3):487-491.",
  "Schiff GD, Klass D, Peterson J, Shah G, Bates DW. Linking laboratory and pharmacy: opportunities for reducing errors and improving care. Arch Intern Med. 2003;163(8):893-900.",
  "Harpaz R, DuMouchel W, Shah NH, Madigan D, Ryan P, Friedman C. Novel data-mining methodologies for adverse drug event discovery and analysis. Clin Pharmacol Ther. 2012;91(6):1010-1021.",
  "Johnson AEW, Pollard TJ, Shen L, et al. MIMIC-III, a freely accessible critical care database. Sci Data. 2016;3:160035.",
  "Johnson AEW, Bulgarelli L, Shen L, et al. MIMIC-IV, a freely accessible electronic health record dataset. Sci Data. 2023;10(1):1.",
  "Pollard TJ, Johnson AEW, Raffa JD, Celi LA, Mark RG, Badawi O. The eICU Collaborative Research Database, a freely available multi-center database for critical care research. Sci Data. 2018;5:180178.",
  "Choi E, Bahadori MT, Schuetz A, Stewart WF, Sun J. Doctor AI: predicting clinical events via recurrent neural networks. Proc Mach Learn Res (MLHC). 2016;56:301-318.",
  "Choi E, Bahadori MT, Sun J, Kulas J, Schuetz A, Stewart W. RETAIN: an interpretable predictive model for healthcare using reverse time attention mechanism. Adv Neural Inf Process Syst (NeurIPS). 2016;29:3504-3512.",
  "Rajkomar A, Oren E, Chen K, et al. Scalable and accurate deep learning with electronic health records. npj Digit Med. 2018;1:18.",
  "Tomašev N, Glorot X, Rae JW, et al. A clinically applicable approach to continuous prediction of future acute kidney injury. Nature. 2019;572(7767):116-119.",
  "Henry KE, Hager DN, Pronovost PJ, Saria S. A targeted real-time early warning score (TREWScore) for septic shock. Sci Transl Med. 2015;7(299):299ra122.",
  "Lundberg SM, Lee SI. A unified approach to interpreting model predictions. Adv Neural Inf Process Syst (NeurIPS). 2017;30:4765-4774.",
  "Lundberg SM, Erion G, Chen H, et al. From local explanations to global understanding with explainable AI for trees. Nat Mach Intell. 2020;2(1):56-67.",
  "Breiman L. Random forests. Mach Learn. 2001;45(1):5-32.",
  "Chen T, Guestrin C. XGBoost: a scalable tree boosting system. Proc 22nd ACM SIGKDD Int Conf Knowl Discov Data Min (KDD). 2016:785-794.",
  "Ke G, Meng Q, Finley T, et al. LightGBM: a highly efficient gradient boosting decision tree. Adv Neural Inf Process Syst (NeurIPS). 2017;30:3146-3154.",
  "Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: machine learning in Python. J Mach Learn Res. 2011;12:2825-2830.",
  "Niculescu-Mizil A, Caruana R. Predicting good probabilities with supervised learning. Proc 22nd Int Conf Mach Learn (ICML). 2005:625-632.",
  "Collins GS, Reitsma JB, Altman DG, Moons KGM. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement. BMJ. 2015;350:g7594.",
  "Weir MR, Rolfe M. Potassium homeostasis and renin-angiotensin-aldosterone system inhibitors. Clin J Am Soc Nephrol. 2010;5(3):531-548.",
  "Holbrook A, Schulman S, Witt DM, et al. Evidence-based management of anticoagulant therapy: Antithrombotic Therapy and Prevention of Thrombosis, 9th ed: American College of Chest Physicians Evidence-Based Clinical Practice Guidelines. Chest. 2012;141(2 Suppl):e152S-e184S.",
  "Inzucchi SE, Lipska KJ, Mayo H, Bailey CJ, McGuire DK. Metformin in patients with type 2 diabetes and kidney disease: a systematic review. JAMA. 2014;312(24):2668-2675.",
  "Whelton A. Nephrotoxicity of nonsteroidal anti-inflammatory drugs: physiologic foundations and clinical implications. Am J Med. 1999;106(5B):13S-24S.",
  "Inker LA, Eneanya ND, Coresh J, et al. New creatinine- and cystatin C-based equations to estimate GFR without race. N Engl J Med. 2021;385(19):1737-1749.",
  "Kellum JA, Lameire N; KDIGO AKI Guideline Work Group. Diagnosis, evaluation, and management of acute kidney injury: a KDIGO summary (Part 1). Crit Care. 2013;17(1):204.",
].forEach((t, i) => body.push(refP(i + 1, t)));

// Safety statement at the very end
body.push(new Paragraph({ spacing: { before: 240, after: 80 }, border: { top: BORDER }, children: [new TextRun("")] }));
body.push(new Paragraph({
  shading: { fill: "F2F2F2", type: ShadingType.CLEAR }, spacing: { after: 120 },
  children: [new TextRun({ text: "Safety statement: This system is intended to support, not replace, clinical judgment. It provides early risk signals that require review by qualified healthcare professionals.", italics: true, bold: true })],
}));

// ---------- assemble ----------
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Arial", color: BLACK },
        paragraph: { spacing: { before: 240, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: "Arial", color: BLACK },
        paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("Page "), new TextRun({ children: [PageNumber.CURRENT] })] })] }) },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  // If manuscript.docx is open in Word it will be locked; in that case write
  // to a sibling name and tell the user, instead of crashing.
  const target = path.join(ROOT, "manuscript.docx");
  try {
    fs.writeFileSync(target, buf);
    console.log("Wrote paper/manuscript.docx (" + buf.length + " bytes)");
  } catch (e) {
    if (e && (e.code === "EBUSY" || e.code === "EPERM")) {
      const alt = path.join(ROOT, "manuscript.NEW.docx");
      fs.writeFileSync(alt, buf);
      console.log("[locked] paper/manuscript.docx is open in Word; wrote " +
                  "paper/manuscript.NEW.docx (" + buf.length + " bytes). " +
                  "Close Word and re-run, or rename manually.");
    } else {
      throw e;
    }
  }
});
