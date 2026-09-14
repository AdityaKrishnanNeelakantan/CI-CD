import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import { getResult } from "../api";
import { ResultsTabs } from "../components/ResultsTabs";

export function ResultsPage() {
  const { resultId } = useParams();
  const resultQuery = useQuery({
    enabled: Boolean(resultId),
    queryKey: ["result", resultId],
    queryFn: () => getResult(resultId!),
    retry: false
  });

  if (resultQuery.isLoading) return <div className="empty">Loading result bundle</div>;
  if (resultQuery.isError || !resultQuery.data) return <div className="alert error">Result bundle could not be loaded.</div>;
  return <ResultsTabs result={resultQuery.data} />;
}
