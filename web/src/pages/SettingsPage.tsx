import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertCircle, CheckCircle2, Save } from "lucide-react";
import { useEffect, useState } from "react";
import { SettingsResponse, getSettings, updateSettings } from "../api";

export function SettingsPage() {
  const queryClient = useQueryClient();
  const settingsQuery = useQuery({ queryKey: ["settings"], queryFn: getSettings, retry: false });
  const [settings, setSettings] = useState<SettingsResponse>({
    generation_mode: "schema_driven",
    default_record_count: 100,
    privacy_level: "standard",
    default_output_format: "csv"
  });
  useEffect(() => {
    if (settingsQuery.data) setSettings(settingsQuery.data);
  }, [settingsQuery.data]);

  const saveMutation = useMutation({
    mutationFn: updateSettings,
    onSuccess: (data) => {
      setSettings(data);
      queryClient.setQueryData(["settings"], data);
    }
  });

  return (
    <section className="panel wide">
      <p className="eyebrow">Settings</p>
      <h2>Generation Defaults</h2>
      <div className="config-grid">
        <label>
          Generation mode
          <select value={settings.generation_mode} onChange={(event) => setSettings({ ...settings, generation_mode: event.target.value })}>
            <option value="schema_driven">Schema driven</option>
            <option value="sample_driven">Sample driven</option>
            <option value="hybrid">Hybrid</option>
          </select>
        </label>
        <label>
          Default records
          <input min={1} type="number" value={settings.default_record_count} onChange={(event) => setSettings({ ...settings, default_record_count: Number(event.target.value) })} />
        </label>
        <label>
          Privacy level
          <select value={settings.privacy_level} onChange={(event) => setSettings({ ...settings, privacy_level: event.target.value })}>
            <option value="standard">Standard</option>
            <option value="strict">Strict</option>
            <option value="air_gapped">Air-gapped</option>
          </select>
        </label>
        <label>
          Default output format
          <select value={settings.default_output_format} onChange={(event) => setSettings({ ...settings, default_output_format: event.target.value })}>
            <option value="csv">CSV</option>
            <option value="parquet">Parquet</option>
            <option value="json">JSON</option>
          </select>
        </label>
      </div>
      <button className="primary" disabled={saveMutation.isPending} onClick={() => saveMutation.mutate(settings)}>
        <Save size={17} />
        Save Settings
      </button>
      {settingsQuery.isLoading ? <div className="empty">Loading settings</div> : null}
      {saveMutation.isSuccess ? (
        <div className="alert success" role="status">
          <CheckCircle2 size={18} />
          <span>Settings saved</span>
        </div>
      ) : null}
      {settingsQuery.isError || saveMutation.isError ? (
        <div className="alert error" role="alert">
          <AlertCircle size={18} />
          <span>Settings could not be loaded or saved.</span>
        </div>
      ) : null}
    </section>
  );
}
