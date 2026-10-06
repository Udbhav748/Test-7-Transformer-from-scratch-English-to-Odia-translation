import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui'

type Translation = { translation: string; strategy: string; source_tokens: number; output_tokens: number }

function TranslatePanel() {
  const [text, setText] = useState('')
  const [strategy, setStrategy] = useState<'greedy' | 'beam'>('greedy')
  const [result, setResult] = useState<Translation | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function run() {
    if (!text.trim()) return
    setBusy(true)
    setError(null)
    try {
      const r = await fetch('/api/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, strategy }),
      })
      const body = await r.json()
      if (!r.ok) throw new Error(body.detail ?? `HTTP ${r.status}`)
      setResult(body)
    } catch (e) {
      setResult(null)
      setError(e instanceof Error ? e.message : 'request failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Translate English to Odia</CardTitle>
        <CardDescription>Runs the trained model locally through the API on port 8000.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={4}
          placeholder="Type an English sentence..."
          className="w-full rounded-lg border border-[var(--border)] bg-transparent p-3 text-sm outline-none focus:ring-2 focus:ring-[var(--border)]"
        />
        <div className="flex flex-wrap items-center gap-3">
          <select
            value={strategy}
            onChange={(e) => setStrategy(e.target.value as 'greedy' | 'beam')}
            className="rounded-lg border border-[var(--border)] bg-transparent px-3 py-2 text-sm"
          >
            <option value="greedy">Greedy (spec)</option>
            <option value="beam">Beam search (bonus)</option>
          </select>
          <button
            onClick={run}
            disabled={busy || !text.trim()}
            className="rounded-lg bg-[var(--primary)] px-4 py-2 text-sm font-medium text-[var(--primary-foreground)] disabled:opacity-50"
          >
            {busy ? 'Translating...' : 'Translate'}
          </button>
        </div>
        {error && <p className="text-sm text-[var(--warning)]">{error}</p>}
        {result && (
          <div className="rounded-lg border border-[var(--border)] p-4">
            <p className="text-lg">{result.translation}</p>
            <p className="mt-2 text-xs text-[var(--muted-foreground)]">
              {result.strategy} · {result.source_tokens} source tokens → {result.output_tokens} output tokens
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default function App() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-8 sm:px-6">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight">Transformer from scratch: English to Odia</h1>
        <p className="text-sm text-[var(--muted-foreground)]">
          Assignment section 5.6 architecture, trained on AI4Bharat Samanantar.
        </p>
      </header>
      <TranslatePanel />
    </div>
  )
}
