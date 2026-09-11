import { useState } from "react";

type DataTablePagination = {
  pageSize?: number;
  totalRows?: number;
};

export function DataTable({
  rows,
  pagination
}: {
  rows: Array<Record<string, unknown>>;
  pagination?: DataTablePagination;
}) {
  const [page, setPage] = useState(0);
  if (!rows.length) return <div className="empty">No rows available</div>;

  const headers = Object.keys(rows[0]);
  const pageSize = Math.max(1, pagination?.pageSize ?? rows.length);
  const pageCount = Math.max(1, Math.ceil(rows.length / pageSize));
  const currentPage = Math.min(page, pageCount - 1);
  const start = currentPage * pageSize;
  const end = Math.min(start + pageSize, rows.length);
  const displayedRows = pagination ? rows.slice(start, end) : rows;
  const totalRows = Math.max(rows.length, pagination?.totalRows ?? rows.length);

  return (
    <div className="data-table">
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {headers.map((header) => (
                <th key={header}>{header}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {displayedRows.map((row, index) => (
              <tr key={start + index}>
                {headers.map((header) => (
                  <td key={header} title={String(row[header] ?? "")}>{String(row[header] ?? "")}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {pagination ? (
        <div className="table-pagination" aria-label="Table pagination">
          <span>
            Showing {start + 1}-{end} of {rows.length} preview rows
            {totalRows > rows.length ? ` (${totalRows.toLocaleString()} generated)` : ""}
          </span>
          <div className="pagination-actions">
            <button className="secondary" disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)} type="button">
              Previous
            </button>
            <span>Page {currentPage + 1} of {pageCount}</span>
            <button className="secondary" disabled={currentPage >= pageCount - 1} onClick={() => setPage(currentPage + 1)} type="button">
              Next
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
