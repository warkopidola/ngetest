# -*- coding: utf-8 -*-
"""Aplikasi Streamlit: Sistem Rekomendasi Buku (Book-Crossing).

Dikonversi dari notebook Recommender_Hazel_009.

Jalankan dengan:
    streamlit run aplikasi.py

Sumber dataset (dipilih di sidebar):
  1. Folder  : taruh BX-Book-Ratings.csv, BX-Books.csv, BX-Users.csv di folder `data/`
               (juga menerima versi .csv.gz atau .zip dengan nama yang sama + ekstensi).
  2. Unggah  : unggah ketiga file langsung lewat browser (.csv, .csv.gz, atau .zip).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Rekomendasi Buku", page_icon="📚", layout="wide")

# ----------------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------------
FOLDER_DATA_DEFAULT = "data"
MIN_RATING_PER_BUKU = 10      # buku minimal punya 10 rating
MIN_RATING_PER_USER = 5       # user minimal memberi 5 rating
SHRINKAGE_CF = 50             # penalti untuk pasangan buku dengan sedikit irisan user
PRIOR_SEGMEN = 1000           # kekuatan prior global pada popularitas per segmen
LABEL_USIA = ["<20", "20-29", "30-44", "45+"]

NAMA_FILE = {
    "ratings": "BX-Book-Ratings.csv",
    "books": "BX-Books.csv",
    "users": "BX-Users.csv",
}
READ_KW = dict(sep=";", encoding="latin-1", escapechar="\\", on_bad_lines="skip")


# ----------------------------------------------------------------------------
# Pembacaan file (dari folder atau dari unggahan)
# ----------------------------------------------------------------------------
def cari_file(folder: str, nama: str):
    """Cari file di folder; menerima .csv, .csv.gz, atau .csv.zip."""
    for kandidat in (nama, nama + ".gz", nama + ".zip"):
        p = Path(folder) / kandidat
        if p.exists():
            return p
    return None


def _baca_csv(src, **kw) -> pd.DataFrame:
    """Baca CSV dari Path (folder) atau UploadedFile (unggahan)."""
    if isinstance(src, Path):
        return pd.read_csv(src, **READ_KW, **kw)  # kompresi dideteksi dari ekstensi
    src.seek(0)
    nama = src.name.lower()
    kompresi = "gzip" if nama.endswith(".gz") else "zip" if nama.endswith(".zip") else None
    return pd.read_csv(src, compression=kompresi, **READ_KW, **kw)


def _kunci_file(src) -> str:
    """Kunci cache ringan: berubah bila file berubah."""
    if isinstance(src, Path):
        st_ = src.stat()
        return f"{src.resolve()}|{st_.st_size}|{st_.st_mtime_ns}"
    return f"{getattr(src, 'file_id', '')}|{src.name}|{src.size}"


# ----------------------------------------------------------------------------
# Membangun model (di-cache supaya tidak dihitung ulang tiap interaksi)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat data dan membangun model, mohon tunggu...")
def bangun_model(_sumber: dict, kunci: str) -> dict:
    # `_sumber` tidak di-hash oleh Streamlit; `kunci` yang menentukan cache.

    # --- Load dataset
    ratings = _baca_csv(_sumber["ratings"])
    books = (
        _baca_csv(
            _sumber["books"],
            usecols=["ISBN", "Book-Title", "Book-Author"],
            low_memory=False,
        )
        .drop_duplicates("ISBN")
        .set_index("ISBN")
    )
    users_df = _baca_csv(_sumber["users"])

    # --- Segmen demografis user (negara | kelompok usia)
    country = users_df["Location"].fillna("").str.split(",").str[-1].str.strip().str.lower()
    top_countries = country[country != ""].value_counts().head(8).index
    country = country.where(country.isin(top_countries), "other")
    age = (
        pd.cut(users_df["Age"], [5, 20, 30, 45, 120], right=False, labels=LABEL_USIA)
        .astype(object)
        .fillna("Unknown")
    )
    segment_of = pd.Series(
        (country + "|" + age).to_numpy(), index=users_df["User-ID"].to_numpy()
    )

    # --- Filter user/buku yang jarang muncul
    df = ratings[(ratings["Book-Rating"] > 0) & ratings["ISBN"].isin(books.index)]
    df = df.drop_duplicates(["User-ID", "ISBN"])
    for _ in range(3):
        df = df[df.groupby("ISBN")["User-ID"].transform("size") >= MIN_RATING_PER_BUKU]
        df = df[df.groupby("User-ID")["ISBN"].transform("size") >= MIN_RATING_PER_USER]

    # --- Matriks user-item (rating dipusatkan per user)
    df = df.assign(
        centered=df["Book-Rating"] - df.groupby("User-ID")["Book-Rating"].transform("mean")
    )
    u, users = pd.factorize(df["User-ID"])
    i, items = pd.factorize(df["ISBN"])
    M = csr_matrix((df["centered"], (u, i)))
    B = csr_matrix((np.ones(len(df)), (u, i)))

    # --- Cosine similarity antar buku + shrinkage berdasarkan jumlah user irisan
    sim = cosine_similarity(M.T).astype(np.float32)
    co = (B.T @ B).toarray().astype(np.float32)
    sim *= co / (co + SHRINKAGE_CF)
    np.fill_diagonal(sim, 0)
    del co

    # --- Popularitas per segmen demografis (dengan smoothing ke popularitas global)
    df["segment"] = df["User-ID"].map(segment_of).fillna("other|Unknown")
    C = pd.crosstab(df["segment"], df["ISBN"]).reindex(columns=items, fill_value=0)
    p_global = C.sum() / C.to_numpy().sum()
    seg_score = (C + PRIOR_SEGMEN * p_global).div(C.sum(axis=1) + PRIOR_SEGMEN, axis=0)

    n_ratings = df["ISBN"].value_counts()
    judul = books.loc[items, "Book-Title"].astype(str)

    return {
        "books": books,
        "segment_of": segment_of,
        "df": df[["User-ID", "ISBN", "Book-Rating"]].copy(),
        "users": users,
        "items": items,
        "M": M,
        "B": B,
        "sim": sim,
        "seg_score": seg_score,
        "p_global": p_global,
        "n_ratings": n_ratings,
        "judul_lower": judul.str.lower().to_numpy(),
        "user_idx": {x: k for k, x in enumerate(users)},
        "item_idx": {x: k for k, x in enumerate(items)},
        "daftar_negara": list(top_countries) + ["other"],
        "n_rating_total": len(df),
        "n_user": df["User-ID"].nunique(),
        "n_buku": df["ISBN"].nunique(),
    }


# ----------------------------------------------------------------------------
# Fungsi rekomendasi
# ----------------------------------------------------------------------------
def _ambil_top(m: dict, scores: np.ndarray, n: int, nama_skor: str) -> pd.DataFrame:
    top = np.argsort(-scores)[:n]
    out = m["books"].loc[m["items"][top], ["Book-Title", "Book-Author"]].copy()
    out[nama_skor] = scores[top]
    out = out[np.isfinite(out[nama_skor])]
    out.index.name = "ISBN"
    return out.rename(columns={"Book-Title": "Judul", "Book-Author": "Penulis"})


def rekomendasi_user(m: dict, user_id: int, n: int = 10):
    """Rekomendasi untuk user yang ada di model (collaborative filtering)."""
    k = m["user_idx"][user_id]
    scores = np.asarray(m["M"][k] @ m["sim"]).ravel()
    scores[m["B"][k].indices] = -np.inf  # buang buku yang sudah dirating
    return _ambil_top(m, scores, n, "Skor")


def rekomendasi_segmen(m: dict, segmen: str | None, n: int = 10):
    """Rekomendasi cold-start: buku populer pada segmen demografis tertentu."""
    if segmen is not None and segmen in m["seg_score"].index:
        scores = m["seg_score"].loc[segmen].to_numpy().copy()
        info = f"Populer di segmen `{segmen}`."
    else:
        scores = m["p_global"].to_numpy().copy()
        info = "Segmen tidak ditemukan, memakai popularitas global."
    return _ambil_top(m, scores, n, "Skor"), info


def cari_kandidat(m: dict, kata_kunci: str, maks: int = 50) -> list:
    """Cari ISBN yang judulnya memuat kata kunci, urut dari yang paling banyak dirating."""
    mask = np.char.find(m["judul_lower"].astype(str), kata_kunci.lower()) >= 0
    cand = list(m["items"][mask])
    cand.sort(key=lambda b: m["n_ratings"][b], reverse=True)
    return cand[:maks]


def buku_serupa(m: dict, isbn: str, n: int = 10) -> pd.DataFrame:
    baris = m["sim"][m["item_idx"][isbn]]
    return _ambil_top(m, baris, n, "Kemiripan")


# ----------------------------------------------------------------------------
# Tampilan
# ----------------------------------------------------------------------------
def tampilkan_tabel(df: pd.DataFrame, nama_skor: str, format_skor: str):
    st.dataframe(
        df,
        width="stretch",
        column_config={nama_skor: st.column_config.NumberColumn(format=format_skor)},
    )


st.title("📚 Sistem Rekomendasi Buku")
st.caption("Item-based collaborative filtering dengan fallback popularitas per segmen demografis (dataset Book-Crossing).")

# --- Sidebar: pilih sumber dataset ---------------------------------------------
sumber = None
with st.sidebar:
    st.header("Pengaturan")
    mode_sumber = st.radio("Sumber dataset", ["Folder (lokal / repo)", "Unggah file"])

    if mode_sumber == "Folder (lokal / repo)":
        folder_data = st.text_input("Folder dataset", value=FOLDER_DATA_DEFAULT)
        ditemukan = {k: cari_file(folder_data, v) for k, v in NAMA_FILE.items()}
        hilang = [NAMA_FILE[k] for k, p in ditemukan.items() if p is None]
        if hilang:
            st.error(
                f"File berikut tidak ditemukan di folder `{folder_data}`:\n\n"
                + "\n".join(f"- {f}" for f in hilang)
            )
        else:
            sumber = ditemukan
    else:
        st.caption("Format yang diterima: .csv, .csv.gz, atau .zip (berisi satu file CSV).")
        unggahan = {
            k: st.file_uploader(NAMA_FILE[k], type=["csv", "gz", "zip"], key=f"unggah_{k}")
            for k in NAMA_FILE
        }
        if all(unggahan.values()):
            sumber = unggahan
        else:
            kurang = [NAMA_FILE[k] for k, f in unggahan.items() if f is None]
            st.info("Masih perlu diunggah: " + ", ".join(kurang))

if sumber is None:
    st.info("Dataset belum lengkap. Lengkapi sumber dataset di sidebar untuk memulai.")
    st.stop()

kunci = "|".join(_kunci_file(sumber[k]) for k in NAMA_FILE)
model = bangun_model(sumber, kunci)

with st.sidebar:
    st.header("Ringkasan data")
    st.metric("Jumlah rating", f"{model['n_rating_total']:,}")
    st.metric("Jumlah user", f"{model['n_user']:,}")
    st.metric("Jumlah buku", f"{model['n_buku']:,}")

tab_user, tab_serupa, tab_tentang = st.tabs(
    ["🎯 Rekomendasi untuk Pengguna", "🔎 Cari Buku Serupa", "ℹ️ Tentang Model"]
)

# --- Tab 1: rekomendasi untuk pengguna ---------------------------------------
with tab_user:
    mode = st.radio(
        "Jenis pengguna",
        ["Pengguna terdaftar (User-ID)", "Pengguna baru (pilih demografi)"],
        horizontal=True,
    )
    n_rek = st.slider("Jumlah rekomendasi", 5, 30, 10, key="n_rek")

    if mode == "Pengguna terdaftar (User-ID)":
        contoh = [int(x) for x in model["users"][:5]]
        user_id = st.number_input(
            "User-ID", min_value=1, value=contoh[0], step=1,
            help="Jika User-ID tidak ada di model, aplikasi memakai segmen demografisnya (cold-start).",
        )
        user_id = int(user_id)
        st.caption(f"Contoh User-ID yang ada di model: {', '.join(map(str, contoh))}. "
                   "Contoh User-ID cold-start: 2.")

        if user_id in model["user_idx"]:
            st.success(f"User {user_id} ditemukan di model: rekomendasi dari collaborative filtering.")
            hasil = rekomendasi_user(model, user_id, n_rek)
            tampilkan_tabel(hasil, "Skor", "%.4f")

            with st.expander("Riwayat rating pengguna ini"):
                df = model["df"]
                riwayat = (
                    df[df["User-ID"] == user_id]
                    .sort_values("Book-Rating", ascending=False)
                    .join(model["books"][["Book-Title", "Book-Author"]], on="ISBN")
                    .rename(columns={"Book-Title": "Judul", "Book-Author": "Penulis",
                                     "Book-Rating": "Rating"})
                    .set_index("ISBN")[["Judul", "Penulis", "Rating"]]
                )
                st.dataframe(riwayat, width="stretch")
        else:
            seg = model["segment_of"].get(user_id)
            st.warning(f"User {user_id} tidak ada di model (cold-start).")
            hasil, info = rekomendasi_segmen(model, seg, n_rek)
            st.info(info)
            tampilkan_tabel(hasil, "Skor", "%.5f")
    else:
        c1, c2 = st.columns(2)
        negara = c1.selectbox("Negara", model["daftar_negara"])
        usia = c2.selectbox("Kelompok usia", LABEL_USIA + ["Unknown"])
        hasil, info = rekomendasi_segmen(model, f"{negara}|{usia}", n_rek)
        st.info(info)
        tampilkan_tabel(hasil, "Skor", "%.5f")

# --- Tab 2: buku serupa --------------------------------------------------------
with tab_serupa:
    kata = st.text_input("Judul buku (sebagian judul sudah cukup)", value="Harry Potter and the Sorcerer")
    n_sim = st.slider("Jumlah buku serupa", 5, 30, 10, key="n_sim")

    if kata.strip():
        kandidat = cari_kandidat(model, kata.strip())
        if not kandidat:
            st.error(f"Judul '{kata}' tidak ditemukan di model.")
        else:
            def label(b):
                return (f"{model['books'].at[b, 'Book-Title']} - {model['books'].at[b, 'Book-Author']} "
                        f"({model['n_ratings'][b]} rating)")

            pilihan = st.selectbox(
                "Pilih buku (diurutkan dari yang paling banyak dirating)",
                kandidat, format_func=label,
            )
            st.write(f"**Query:** {model['books'].at[pilihan, 'Book-Title']} "
                     f"- {model['books'].at[pilihan, 'Book-Author']}")
            tampilkan_tabel(buku_serupa(model, pilihan, n_sim), "Kemiripan", "%.4f")

# --- Tab 3: tentang model --------------------------------------------------------
with tab_tentang:
    st.markdown(
        f"""
**Pra-pemrosesan.** Hanya rating eksplisit (> 0) yang dipakai. Buku dengan kurang dari
{MIN_RATING_PER_BUKU} rating dan user dengan kurang dari {MIN_RATING_PER_USER} rating dibuang secara
iteratif (3 putaran). Rating dipusatkan per user (dikurangi rata-rata rating user tersebut).

**Kemiripan antar buku.** Cosine similarity pada matriks user-item yang sudah dipusatkan,
lalu diberi shrinkage berdasarkan banyaknya user yang menilai kedua buku:
"""
    )
    st.latex(r"s'_{ij} = s_{ij} \cdot \frac{n_{ij}}{n_{ij} + %d}" % SHRINKAGE_CF)
    st.markdown(
        """
**Skor rekomendasi** untuk user *k* adalah hasil kali vektor rating terpusat user dengan matriks
kemiripan; buku yang sudah dirating user dikeluarkan.

**Cold-start.** User yang tidak ada di model direkomendasikan buku populer di segmen
demografisnya (negara | kelompok usia). Popularitas segmen dihaluskan terhadap popularitas global:
"""
    )
    st.latex(
        r"\text{skor}_{s,b} = \frac{n_{s,b} + %d\, p_b}{n_s + %d}" % (PRIOR_SEGMEN, PRIOR_SEGMEN)
    )
