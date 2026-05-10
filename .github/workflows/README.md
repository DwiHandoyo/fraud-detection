# CI/CD

Memenuhi spec IF5251 nomor **Operations 4a**.

## Jobs (urut)

```
lint ─┬─ test-ml-service ─┐
      └─ test-data-validation ─┴─ build-images
```

| Job | Apa yang dilakukan |
|-----|-------------------|
| `lint` | ruff check pada semua subfolder Python (gating ringan; ignore E501, F401) |
| `test-ml-service` | install deps + seed feature store + `pytest tests/` (20 tests) |
| `test-data-validation` | install deps + seed data + `validate_dataset.py` (40 expectations) |
| `build-images` | build Dockerfile ml-service & ui (validasi Dockerfile, no push) |

## Trigger

- `push` ke `main`
- `pull_request` apapun

## Catatan

- Belum push image ke registry. Untuk production deploy, tambah step
  `docker login` + `docker push` ke GHCR/ECR (butuh secret `DOCKERHUB_TOKEN`
  atau equivalent).
- `feast materialize` saat CI butuh Python 3.11 dan ~30 detik. Cache pip
  digunakan untuk mempercepat run berikutnya.
- `lint` saat ini soft-fail (`|| true`). Setelah codebase di-format konsisten,
  hapus untuk strict gating.

## Local equivalent

Tidak butuh GitHub untuk verify — jalankan local:

```bash
# Lint
ruff check ml-service/ feature-store/ data-validation/ ui/

# Tests
cd ml-service && pytest tests/ -v

# Validation
cd ../data-validation && python validate_dataset.py

# Image builds
cd .. && docker build -t fraud-ml:local ml-service/
docker build -t fraud-ui:local ui/
```
