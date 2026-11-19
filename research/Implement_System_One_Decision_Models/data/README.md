# Synthetic Support Tickets

These 72 ticket texts and route labels were authored for this tutorial. No
external support dataset was copied. The data is released under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

The `dev` and `test` values in `tickets.csv` are fixed. Each split has 36
cases, balanced across the three routes. The label is the team responsible
for the ticket's main requested resolution. Mixed-topic cases say which
resolution takes priority. The cases are short and synthetic, so they do not
represent a production ticket distribution. We cannot rule out overlap with
the model's undisclosed training examples.

| Route | Meaning |
| :--- | :--- |
| `billing` | Charges, payments, refunds, invoices |
| `shipping` | Delivery, tracking, courier problems |
| `technical` | App, account, website, or product faults |
