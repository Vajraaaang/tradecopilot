# Design QA

- Color and material reference: `/var/folders/8v/dx_7kmwd3hvcpvtywqbw6_t40000gn/T/TemporaryItems/NSIRD_screencaptureui_TwhkOW/Screenshot 2026-08-10 at 3.45.35 AM.png`
- Trading layout reference: `/var/folders/8v/dx_7kmwd3hvcpvtywqbw6_t40000gn/T/TemporaryItems/NSIRD_screencaptureui_Vw083u/Screenshot 2026-08-10 at 3.08.30 AM.png`
- Implementation evidence: `docs/design-qa-implementation.jpg`
- Browser viewport: 1728 x 906 CSS pixels
- Tested state: completed YXT synthetic replay at `REENTRY_WATCH`, with 1-minute candles and a flat-after-exit replay position

## Visual comparison

The latest color reference and the implementation were rendered together in
the same Chrome inspection output. The surrounding app carries over the white
foreground, pale violet/blue field, compact controls, black typography, and
restrained purple accents. The chart itself follows TradingView's light-chart
hierarchy: white canvas, quiet gray grid/scales, blue/green/red analytical
accents, compact toolbar, and bottom navigation.

The background is approximately 90% white with a restrained animated lavender
glow concentrated near the upper-right and lower edge. Navigation, positions,
screener, copilot, the bottom risk strip, and the analysis drawer share one
translucent glass recipe. The chart is intentionally opaque white for scale,
grid, candle, and indicator legibility.

## Interaction evidence

- Analysis desk opened and exposed the deterministic session brief, then closed.
- Indicators opened the visible-studies menu containing Volume, VWAP, EMA9,
  EMA20, RSI14, and MACD.
- Selecting `5m` changed chart status to `5m · extended_hours`; selecting `1m`
  restored the detailed view.
- Candle and line chart modes both rendered; logarithmic scale toggled without
  changing deterministic prices; two zoom-in actions reduced the visible bar
  window; Reset restored all 29 bars.
- The Full screen control entered browser fullscreen for the complete app and
  changed to `Exit full screen`; it then returned to the normal viewport.
- Loading `SBFM` failed closed with an explicit read-only-provider limitation;
  the current YXT replay remained intact.
- The data badge reads `REPLAY DATA`, not `LOCAL LIVE`; the top bar separately
  labels the regular market closed and shows current Pacific time.
- The health endpoint exposes `write_capabilities: false`; `/api/order` is not a
  route and returns 404 in automated coverage.
- GPT controls are disabled when `OPENAI_API_KEY` is absent and the setup note
  is visible. The interface does not disguise deterministic canned text as GPT.

## Findings and iterations

1. P1: an intermediate pass made the screener the only opaque right-rail panel.
   The later user direction superseded that treatment, so all three rail panels
   now share the same translucent material and internal header opacity.
2. P1: `LOCAL LIVE` could be mistaken for live market data while replaying after
   hours. It now reads `REPLAY DATA`; replay state and wall-clock market status
   remain separate.
3. P2: the animated backdrop was muted by a workspace veil. The veil was reduced
   while preserving contrast on chart and rail content.
4. P1: the first gradient pass was visibly too purple against the latest crop
   supplied by the user. The base was changed to white and the violet fields
   were reduced in opacity, area, and saturation while retaining slow motion.
5. No clipped panels, broken controls, hidden data limitations, or misleading
   live-state labels remained in the final 1728 x 906 viewport.
6. P1: adding chart controls initially overloaded the three-column footer and
   was perceived as losing fullscreen. The footer now has dedicated range,
   navigation, interval, and status columns, while the app shell is explicitly
   edge-to-edge and fullscreen targets the whole document.
7. P1: the dark chart conflicted with the requested light TradingView direction.
   Header, canvas, indicator panes, axes, tool menus, and footer now use a
   purpose-built light palette rather than generic glass cards.
8. P1: the light chart, dark chat message, pill actions, and heavier glass rail
   read as separate design systems. The final pass standardizes a 7px radius,
   one border/shadow recipe, Avenir-based typography, a single violet accent,
   compact rectangular actions, and a light assistant message with a violet
   keyline.

final result: passed
