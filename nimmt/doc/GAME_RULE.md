# Simplified 6 nimmt! Rules

This implementation is a compact single-round version for bot battles and the
local browser UI.

## Setup

- Players: `2..6`.
- Deck: `1..10N`, where `N` is the player count.
- Each player receives `10` cards.
- There are no extra initial row cards. The first four resolved cards open the
  four public rows.

## Turn Flow

Each turn all players choose one card at the same time. Chosen cards are then
resolved from low to high.

If a card is lower than every current row top card, the action must also choose
which row to take, for example `PLAY_3_TAKE_2`.

## Row Placement

- A card is placed on the row whose top card is lower than the played card and
  closest to it.
- If the target row already has five cards, the player takes that row's bull
  heads and the played card starts a new row.
- If the played card is lower than all row tops, the chosen row is taken and the
  played card starts that row.

## Bull Heads

- `55` is worth `7`.
- Other multiples of `11` are worth `5`.
- Other multiples of `10` are worth `3`.
- Other multiples of `5` are worth `2`.
- All other cards are worth `1`.

## Result

After all players have played all ten cards, the player or players with the
lowest bull-head score win.
