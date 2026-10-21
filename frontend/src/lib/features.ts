/** The 52 TEP measurement features in the model's canonical order
 *  (mirrors production_pulse.data.FEATURE_COLUMNS). The API's GET /model/status
 *  returns the authoritative list; this copy is used until it has loaded. */
export const XMEAS_COLUMNS: readonly string[] = Array.from({ length: 41 }, (_, i) => `xmeas_${i + 1}`);
export const XMV_COLUMNS: readonly string[] = Array.from({ length: 11 }, (_, i) => `xmv_${i + 1}`);
export const FEATURE_COLUMNS: readonly string[] = [...XMEAS_COLUMNS, ...XMV_COLUMNS];

/** Columns of the original TEP files: allowed in CSV uploads, never sent as features. */
export const TEP_METADATA_COLUMNS: readonly string[] = ["faultNumber", "simulationRun", "sample"];
/** Optional helper columns: allowed in CSV uploads, used for labels only. */
export const AUX_COLUMNS: readonly string[] = ["sample_id", "timestamp", "time", "datetime"];
export const TIMESTAMP_COLUMNS: readonly string[] = ["timestamp", "time", "datetime"];

export type FeatureGroup = "XMEAS" | "XMV";

export function featureGroup(name: string): FeatureGroup {
  return name.startsWith("xmv_") ? "XMV" : "XMEAS";
}

export function featureGroupLabel(group: FeatureGroup): string {
  return group === "XMV" ? "Manipulated variables (XMV)" : "Process measurements (XMEAS)";
}
