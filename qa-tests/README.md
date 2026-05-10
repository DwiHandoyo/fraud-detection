# qa-tests — Adversarial + load testing

Memenuhi spec IF5251 nomor **QA 3b** (adversarial input) dan **QA 3c** (load
testing).

## Files

| File | Tujuan | Output |
|------|--------|--------|
| `adversarial.py` | Input perturbation, ukur flip rate | `reports/adversarial_report.json` |
| `load_test.py` | Locust scenarios untuk RPS / latency | `reports/load_test_report.html` |

## Adversarial test

Untuk 23 transaksi sample, perturb 6 fitur numerik (TransactionAmt, id_02,
id_05, id_11, id_19, id_20) dengan ±1%, ±5%, ±10% dan hitung berapa prediksi
yang flip (legit ↔ fraud).

```bash
# ml-service harus jalan
cd ml-service && ../playground/.venv/bin/python -m uvicorn app.main:app --port 8000 &

# di terminal lain
cd qa-tests
../playground/.venv/bin/python adversarial.py --base-url http://localhost:8000
```

Output mirip:

```
=== flip rate per feature × perturbation ===
feature        -10%   -5%   -1%   +1%   +5%  +10%
TransactionAmt 0.04  0.00  0.00  0.00  0.00  0.04
id_02          0.13  0.04  0.00  0.00  0.04  0.09
...
```

Interpretasi:
- **Flip rate tinggi (>20%)** → model rentan terhadap small perturbation, tidak robust
- **Flip rate rendah (<5%)** → model stabil
- Asymmetric (negative perturbation flip > positive) → model berat sebelah

## Load test

Locust scenario simulates 50 concurrent users hitting `/predict_by_id`
(weight 7), `/explain_by_id` (weight 2), `/health` (weight 1).

```bash
# ml-service harus jalan
cd qa-tests
locust -f load_test.py --host http://localhost:8000 \
    --users 50 --spawn-rate 10 --run-time 60s --headless \
    --html reports/load_test_report.html
```

Metrik yang dihasilkan locust:
- Throughput (RPS)
- Latency p50 / p95 / p99
- Error rate per endpoint
- Failures grouped by exception/HTTP code

Untuk spike test (200 users selama 10s):
```bash
locust -f load_test.py --host http://localhost:8000 \
    --users 200 --spawn-rate 200 --run-time 10s --headless \
    --html reports/load_test_spike.html
```

## Untuk demo

Slide presentation:
1. Tampilkan adversarial table — sebagai bukti model robust (atau tidak)
2. Tampilkan locust HTML — grafik RPS vs latency
3. Tunjukkan p99 latency — sebanding dengan SLO yang dijanjikan di spec
   (latency budget biasanya 100ms untuk fraud detection real-time)

## Limitasi

- Adversarial perturbation di sini sederhana (relative scaling). Untuk
  attack yang lebih realistis, pakai FGSM (Fast Gradient Sign Method) atau
  PGD — butuh akses ke gradient model (LightGBM tree-based, tidak
  differentiable, jadi gradient attack tidak applicable; tapi pakai
  surrogate neural net mungkin)
- Load test single-machine. Untuk distributed load, jalankan locust dengan
  multiple workers
