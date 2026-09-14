import { Download } from "lucide-react";
import { ArtifactFileMetadata } from "../api";

export function DownloadList({
  artifacts,
  onDownload
}: {
  artifacts: ArtifactFileMetadata[];
  onDownload: (artifact: ArtifactFileMetadata) => void;
}) {
  if (!artifacts.length) return <div className="empty">No generated files are available yet</div>;
  return (
    <div className="download-list">
      {artifacts.map((artifact) => (
        <div className="download-row" key={artifact.id}>
          <div>
            <strong>{artifact.filename ?? artifact.name}</strong>
            <span>
              {artifact.kind ?? artifact.role ?? "artifact"} - {artifact.content_type ?? artifact.media_type}
            </span>
          </div>
          {artifact.downloadable && artifact.download_url ? (
            <button className="icon-button" onClick={() => onDownload(artifact)} title="Download">
              <Download size={18} />
            </button>
          ) : (
            <span className="muted">Metadata only</span>
          )}
        </div>
      ))}
    </div>
  );
}
