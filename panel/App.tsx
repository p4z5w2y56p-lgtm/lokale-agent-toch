import data from './data.json'

type Row = { regime: string; risk: string; agent: string; decide_every: number | null; total_return: number; buy_hold_return: number; max_drawdown: number; trades: number; proposals_rejected: number; proposals_total: number }
const rows = data.sweep.rows as Row[]
const regimes = data.sweep.regimes as Record<string, { start_ts: number; end_ts: number; move: number }>
const pct = (x: number) => `${x >= 0 ? '+' : ''}${(x * 100).toFixed(2)}%`
const day = (t: number) => new Date(t * 1000).toISOString().slice(0, 10)
const tone = (x: number) => (x > 0.0005 ? 'text-emerald-400' : x < -0.0005 ? 'text-rose-400' : 'text-zinc-400')

const cols: { label: string; pick: (r: Row) => boolean }[] = [
  { label: 'Baseline', pick: r => r.agent === 'baseline' && r.risk === 'normal' },
  { label: 'Haiku /3', pick: r => r.agent === 'haiku' && r.risk === 'normal' && r.decide_every === 3 },
  { label: 'Haiku /6', pick: r => r.agent === 'haiku' && r.risk === 'normal' && r.decide_every === 6 },
  { label: 'Haiku /12', pick: r => r.agent === 'haiku' && r.risk === 'normal' && r.decide_every === 12 },
  { label: 'Haiku /6 strict', pick: r => r.agent === 'haiku' && r.risk === 'strict' },
]

const log = [
  ['synthetic', 'decide every 8', '0 trades, +0.00% vs B&H -6.19%'],
  ['synthetic', 'decide every 3', 'test -0.09% vs B&H -6.19%, 4 trades, 0 wins'],
  ['real, 4 regimes', 'capability sweep', '20 Haiku runs, 720 calls, 1.28 EUR'],
]

export default function App() {
  const used = data.spent / data.budget
  const haiku = rows.filter(r => r.agent === 'haiku')
  const beatsBH = haiku.filter(r => r.total_return > r.buy_hold_return).length
  return (
    <div className="min-h-screen bg-[#0d0f0c] text-zinc-200 font-mono px-4 py-6 md:px-10">
      <header className="flex flex-wrap items-baseline justify-between gap-2 border-b border-zinc-800 pb-4">
        <div>
          <h1 className="text-4xl font-black tracking-tight text-lime-300">OMIN</h1>
          <p className="text-xs text-zinc-500">crypto agent · Haiku 4.5 brain · paper + Base Sepolia testnet only</p>
        </div>
        <span className="text-xs border border-amber-500/50 text-amber-300 px-2 py-1">STAGE 0 · REPLAY · no real money</span>
      </header>

      <section className="grid md:grid-cols-3 gap-4 mt-6">
        <div className="md:col-span-2 border border-zinc-800 p-4">
          <h2 className="text-xs uppercase text-zinc-500">Model spend (estimate)</h2>
          <p className="text-3xl mt-1">€{data.spent.toFixed(2)} <span className="text-base text-zinc-500">/ €{data.budget.toFixed(2)}</span></p>
          <div className="h-3 bg-zinc-800 mt-3"><div className={`h-3 ${used > 0.8 ? 'bg-rose-500' : 'bg-lime-400'}`} style={{ width: `${Math.min(used, 1) * 100}%` }} /></div>
          <p className="text-xs text-zinc-500 mt-2">Hard stop at the cap: no call goes out once €{data.budget.toFixed(2)} is reached. Real bill: console.anthropic.com → Settings → Billing.</p>
          <table className="text-xs mt-3 w-full">
            <tbody>{Object.entries(data.usage).map(([k, v]) => (
              <tr key={k} className="border-t border-zinc-800"><td className="py-1">{k}</td><td>{v.calls || '–'} calls</td><td>{(v.in / 1000).toFixed(0)}k in / {(v.out / 1000).toFixed(0)}k out</td><td className="text-right">€{v.eur.toFixed(2)}</td></tr>
            ))}</tbody>
          </table>
        </div>
        <div className="border border-zinc-800 p-4">
          <h2 className="text-xs uppercase text-zinc-500">Wallet</h2>
          <p className="text-sm mt-1">Base Sepolia (84532)</p>
          <p className="text-[11px] break-all text-zinc-400">0xEcDE4Ef5bE98227926cc677d2c0b37B61e3AC92c</p>
          <p className="text-xs mt-2">Wrapped 0.00005 ETH → WETH. Cap 0.01 ETH/tx. Next swap waits for Matt's "send".</p>
          <p className="text-xs text-zinc-500 mt-2">Read only. This page cannot sign anything.</p>
        </div>
      </section>

      <section className="mt-6 border border-zinc-800 p-4">
        <h2 className="text-xs uppercase text-zinc-500">Where Haiku gets too weak · real Binance 1h, ETH + BTC, 10-day windows</h2>
        <p className="text-sm mt-2">Haiku beat buy-and-hold in {beatsBH}/{haiku.length} runs, but only by sitting in cash while the market fell. In the bull window it made ~+1% while holding made +28%. It never found an edge on its own.</p>
        <div className="overflow-x-auto mt-3">
          <table className="text-xs w-full min-w-[640px]">
            <thead><tr className="text-zinc-500 text-left"><th className="py-1">Regime</th><th>Buy &amp; hold</th>{cols.map(c => <th key={c.label}>{c.label}</th>)}</tr></thead>
            <tbody>{Object.entries(regimes).map(([name, g]) => (
              <tr key={name} className="border-t border-zinc-800">
                <td className="py-2">{name}<div className="text-[10px] text-zinc-500">{day(g.start_ts)} → {day(g.end_ts)}</div></td>
                <td className={tone(rows.find(r => r.regime === name)!.buy_hold_return)}>{pct(rows.find(r => r.regime === name)!.buy_hold_return)}</td>
                {cols.map(c => { const r = rows.find(x => x.regime === name && c.pick(x))!; return <td key={c.label} className={tone(r.total_return)}>{pct(r.total_return)}<div className="text-[10px] text-zinc-500">{r.trades} trades</div></td> })}
              </tr>
            ))}</tbody>
          </table>
        </div>
      </section>

      <section className="mt-6 border border-zinc-800 p-4">
        <h2 className="text-xs uppercase text-zinc-500">Training log</h2>
        <ul className="text-xs mt-2 space-y-1">{log.map(l => <li key={l[1]}><span className="text-lime-300">{l[0]}</span> · {l[1]} · {l[2]}</li>)}</ul>
      </section>
      <footer className="text-[10px] text-zinc-600 mt-6">snapshot {data.generated} · regenerate with agent-trader sweep</footer>
    </div>
  )
}
