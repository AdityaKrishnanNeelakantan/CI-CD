import { FileUp } from "lucide-react";
import { ChangeEvent, ReactNode } from "react";

export function UploadPanel({
  accept,
  busy,
  children,
  label,
  onFile
}: {
  accept: string;
  busy?: boolean;
  children?: ReactNode;
  label: string;
  onFile: (file: File) => void;
}) {
  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) onFile(file);
    event.target.value = "";
  }

  return (
    <div className="upload-panel">
      <label className="file-drop">
        <FileUp size={22} />
        <span>{busy ? "Uploading..." : label}</span>
        <input aria-label={label} type="file" accept={accept} onChange={handleChange} />
      </label>
      {children}
    </div>
  );
}
