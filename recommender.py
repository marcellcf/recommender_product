# recommender.py# recommender.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import time
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import random

# Set page config
st.set_page_config(
    page_title="🏦 BANK BTN Product Recommender",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 1. DATA LOADER ====================
class DataLoader:
    def __init__(self):
        self.users_df = None
        self.transactions_df = None

    def load_data(self, users_path, transactions_path):
        """Load data"""
        with st.spinner("📂 Loading data..."):
            start = time.time()
            
            self.users_df = pd.read_csv(users_path)
            self.transactions_df = pd.read_csv(transactions_path)
            
            elapsed = time.time() - start
            
        st.success(f"✅ Loaded: {len(self.users_df):,} users, {len(self.transactions_df):,} transactions ({elapsed:.1f}s)")
        return self.users_df, self.transactions_df

# ==================== 2. ULTRA OPTIMIZED CBF ====================
class UltraOptimizedCBF:
    def __init__(self):
        self.user_features = None
        self.product_profiles = None
        self.cbf_scores_cache = None
        self.user_cif_to_idx = None
        self.product_to_idx = None

    def fit(self, users_df, transactions_df):
        """Build CBF model dengan FULL caching"""
        with st.spinner("🧠 Building CBF model..."):
            start_time = time.time()

            self.user_features = self._prepare_features(users_df)
            self.product_profiles = self._build_product_profiles(users_df, transactions_df)
            self._precompute_all_cbf_scores()

            elapsed = time.time() - start_time
        return self

    def _prepare_features(self, users_df):
        """Prepare user features"""
        features = users_df.copy()

        # Gender encoding
        features['GENDER_CODE'] = features['GENDER'].map({'M': 1, 'F': 2}).fillna(1)

        # Occupation encoding (18 jenis)
        occ_mapping = {
            'KARYAWAN': 1, 'WIRASWASTA': 2, 'MANUFAKTUR/INDUSTRI': 3,
            'TIDAK MENCANTUMKAN': 4, 'MAHASISWA': 5, 'PERDAGANGAN': 6,
            'IBU RUMAH TANGGA': 7, 'GURU': 8, 'BURUH': 9,
            'PENDIDIKAN': 10, 'PELAJAR': 11, 'PNS': 12,
            'TEKNOLOGI': 13, 'TRANSPORTASI DARAT': 14, 'PERBANKAN': 15,
            'PROPERTI/KONSTRUKSI': 16, 'MARKETING': 17, 'LAINNYA': 18
        }
        features['OCCUPATION_CODE'] = features['OCCUPATION'].map(occ_mapping).fillna(1)

        # Balance encoding (6 kategori)
        balance_order = ['1-5jt', '5-10jt', '10-25jt', '25-50jt', '50-100jt', '>100jt']
        balance_map = {bal: i+1 for i, bal in enumerate(balance_order)}
        features['BALANCE_CODE'] = features['RANGE_SALDO'].map(lambda x: balance_map.get(x, 1))

        # Age standardization
        age_mean = features['AGE'].mean()
        age_std = features['AGE'].std()
        if age_std == 0:
            age_std = 1.0
        features['AGE_STANDARDIZED'] = (features['AGE'] - age_mean) / age_std

        return features

    def _build_product_profiles(self, users_df, transactions_df):
        """Build product profiles"""
        user_features = self._prepare_features(users_df)

        merged = pd.merge(
            transactions_df[['CIF', 'TRANSACTION_BILLER']],
            user_features[['CIF', 'GENDER_CODE', 'OCCUPATION_CODE', 'BALANCE_CODE', 'AGE_STANDARDIZED']],
            on='CIF'
        )

        product_profiles = {}
        for product in merged['TRANSACTION_BILLER'].unique():
            product_data = merged[merged['TRANSACTION_BILLER'] == product]
            if len(product_data) >= 1:
                product_profiles[product] = product_data[
                    ['GENDER_CODE', 'OCCUPATION_CODE', 'BALANCE_CODE', 'AGE_STANDARDIZED']
                ].mean().to_dict()

        return product_profiles

    def _precompute_all_cbf_scores(self):
        """Pre-compute SEMUA CBF scores dengan matrix operations"""
        # Create mappings
        self.user_cif_to_idx = {cif: idx for idx, cif in enumerate(self.user_features['CIF'])}
        self.product_to_idx = {product: idx for idx, product in enumerate(self.product_profiles.keys())}

        # Prepare user vectors matrix
        user_vectors = []
        for _, user in self.user_features.iterrows():
            user_vectors.append([
                user['GENDER_CODE'],
                user['OCCUPATION_CODE'],
                user['BALANCE_CODE'],
                user['AGE_STANDARDIZED']
            ])
        user_matrix = np.array(user_vectors, dtype=np.float32) + 1e-10

        # Prepare product vectors matrix
        product_list = list(self.product_profiles.keys())
        product_vectors = []
        for product in product_list:
            profile = self.product_profiles[product]
            product_vectors.append([
                profile['GENDER_CODE'],
                profile['OCCUPATION_CODE'],
                profile['BALANCE_CODE'],
                profile['AGE_STANDARDIZED']
            ])
        product_matrix = np.array(product_vectors, dtype=np.float32) + 1e-10

        # Compute cosine similarity matrix
        user_norms = np.linalg.norm(user_matrix, axis=1, keepdims=True)
        user_norms[user_norms < 1e-5] = 1e-5

        product_norms = np.linalg.norm(product_matrix, axis=1, keepdims=True)
        product_norms[product_norms < 1e-5] = 1e-5

        dot_products = np.dot(user_matrix, product_matrix.T)
        norm_products = user_norms * product_norms.T
        similarity_matrix = dot_products / norm_products

        # Clip to [0, 1]
        similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)

        # Store cache
        self.cbf_scores_cache = similarity_matrix

    def get_cbf_score_fast(self, cif, product):
        """ULTRA FAST: Get CBF score from cache (O(1))"""
        if cif not in self.user_cif_to_idx or product not in self.product_to_idx:
            return 0.0

        user_idx = self.user_cif_to_idx[cif]
        product_idx = self.product_to_idx[product]

        return float(self.cbf_scores_cache[user_idx, product_idx])

# ==================== 3. ITEM-BASED CF (SCALABLE) ====================
class ItemBasedCF:
    """ITEM-BASED Collaborative Filtering - SCALABLE untuk 4j users"""

    def __init__(self, min_common_users=3):
        self.min_common_users = min_common_users
        self.user_product_matrix = None
        self.product_similarity = None
        self.product_to_idx = None
        self.user_id_to_idx = None

    def fit(self, transactions_df):
        """Build item-item similarity matrix (SCALABLE)"""
        with st.spinner("🔍 Building ITEM-BASED CF..."):
            start_time = time.time()

            # Build user-product matrix
            self.user_product_matrix = pd.crosstab(
                transactions_df['CIF'],
                transactions_df['TRANSACTION_BILLER']
            ).clip(upper=1)

            # Create fast lookup dictionaries
            self.user_id_to_idx = {cif: idx for idx, cif in enumerate(self.user_product_matrix.index)}

            products = self.user_product_matrix.columns.tolist()
            self.product_to_idx = {p: i for i, p in enumerate(products)}

            n_products = len(products)

            # Convert to numpy untuk perhitungan cepat
            matrix = self.user_product_matrix.values.astype(np.float32).T

            # Vectorized cosine similarity calculation
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized = matrix / norms

            self.product_similarity = np.dot(normalized, normalized.T)
            np.fill_diagonal(self.product_similarity, 1.0)

            # Filter out similarities with too few common users
            for i in range(n_products):
                for j in range(i+1, n_products):
                    common_users = np.sum((matrix[i] > 0) & (matrix[j] > 0))
                    if common_users < self.min_common_users:
                        self.product_similarity[i, j] = 0.0
                        self.product_similarity[j, i] = 0.0

            elapsed = time.time() - start_time
        return self

    def get_cf_score(self, cif, product):
        """Get CF score berdasarkan item similarity"""
        if product not in self.product_to_idx:
            return 0.0

        if cif not in self.user_id_to_idx:
            return 0.0

        user_idx = self.user_id_to_idx[cif]

        # Jika sudah transaksi produk ini, return 0.9
        if self.user_product_matrix.iloc[user_idx][product] > 0:
            return 0.9

        target_product_idx = self.product_to_idx[product]

        # Get products yang sudah dibeli user ini
        user_vector = self.user_product_matrix.iloc[user_idx]
        purchased_products = user_vector[user_vector > 0].index.tolist()

        if not purchased_products:
            return 0.0

        # Hitung rata-rata similarity ke produk yang sudah dibeli
        total_similarity = 0.0
        count = 0

        for purchased_product in purchased_products:
            if purchased_product in self.product_to_idx:
                purchased_idx = self.product_to_idx[purchased_product]
                similarity = self.product_similarity[target_product_idx, purchased_idx]

                if similarity > 0.1:
                    total_similarity += similarity
                    count += 1

        if count > 0:
            score = total_similarity / count
            return min(score, 1.0)

        return 0.0

# ==================== 4. PRODUCT RECOMMENDER ====================
class ProductRecommender:
    def __init__(self, cf_weight=0.3, cbf_weight=0.7):
        self.cf_weight = cf_weight
        self.cbf_weight = cbf_weight
        self.cf_model = ItemBasedCF()
        self.cbf_model = UltraOptimizedCBF()
        self.users_df = None
        self.transactions_df = None
        self.is_trained = False

    def fit(self, users_df, transactions_df):
        """Train models"""
        with st.spinner("🚀 Training recommender..."):
            start_time = time.time()

            self.users_df = users_df
            self.transactions_df = transactions_df

            self.cf_model.fit(transactions_df)
            self.cbf_model.fit(users_df, transactions_df)
            
            self.is_trained = True
            
            elapsed = time.time() - start_time
        return self

    def get_all_users_for_product(self, product_name, min_score=0.01):
        """Get ALL users with scores"""
        all_user_scores = []
        
        # Pre-compute CF scores untuk active users
        active_users = self.users_df[~self.users_df['IS_COLD']]
        cf_scores = {}

        active_cifs = active_users['CIF'].tolist()
        for cif in active_cifs:
            cf_scores[cif] = self.cf_model.get_cf_score(cif, product_name)

        # Progress bar dan satu pesan loading
        progress_bar = st.progress(0)
        total_users = len(self.users_df)
        
        # Pesan loading tunggal yang akan diupdate
        progress_message = st.empty()
        initial_messages = [
            "🕒 **Nunggu 3-5 menit ya!** Lagi analisis customer...",
            "⏳ **Butuh 3-7 menit nih!** Sabar ya, hasilnya worth it kok!",
            "☕ **Sambil nunggu 3-10 menit**, boleh nyemil dulu!",
            "🔍 **Tunggu 4-8 menit ya!** Lagi proses ribuan data...",
            "🎯 **Sedang cari customer terbaik**, butuh 3-9 menit!"
        ]
        progress_message.info(random.choice(initial_messages))
        
        for i, (_, user) in enumerate(self.users_df.iterrows(), 1):
            cif = user['CIF']
            is_cold = user['IS_COLD']

            cbf_score = self.cbf_model.get_cbf_score_fast(cif, product_name)

            if is_cold:
                cf_score = 0.0
                user_status = 'COLD'
            else:
                cf_score = cf_scores.get(cif, 0.0)
                user_status = 'ACTIVE'

            # Calculate final score
            final_score = (cf_score * self.cf_weight) + (cbf_score * self.cbf_weight)

            if final_score >= min_score:
                all_user_scores.append({
                    'CIF': cif,
                    'FINAL_SCORE': round(final_score, 4),
                    'USER_STATUS': user_status,
                    'PRODUCT': product_name,
                    'AGE': user['AGE'],
                    'GENDER': user['GENDER'],
                    'OCCUPATION': user['OCCUPATION'],
                    'RANGE_SALDO': user['RANGE_SALDO']
                })
            
            # Update progress bar dan pesan setiap 10%
            if i % (total_users // 10) == 0 or i == total_users:
                progress_percent = int((i / total_users) * 100)
                progress_bar.progress(progress_percent / 100)
                
                # Update pesan progress
                if progress_percent < 100:
                    progress_messages = [
                        f"📈 **{progress_percent}% complete!** Sedang proses {i} dari {total_users} customer...",
                        f"🔥 **{progress_percent}% done!** Masih ada {total_users - i} customer lagi...",
                        f"⚡ **{progress_percent}% selesai!** Sabar ya, lagi jalan nih!",
                        f"🎪 **{progress_percent}% processed!** Masih seru nih analisisnya!",
                        f"🚦 **{progress_percent}% complete!** Tunggu bentar lagi ya!"
                    ]
                    progress_message.info(random.choice(progress_messages))

        progress_bar.empty()
        progress_message.empty()
        
        # Pesan selesai
        completion_messages = [
            "🎉 **Selesai!** Proses analisis berhasil!",
            "✅ **Done!** Hasilnya sudah keluar!",
            "✨ **Analysis complete!** Silakan lihat hasilnya!",
            "🏁 **Finish!** Proses berjalan lancar!",
            "🥳 **Berhasil!** Data siap dilihat!"
        ]
        st.success(random.choice(completion_messages))
        
        # Sort results
        all_user_scores.sort(key=lambda x: x['FINAL_SCORE'], reverse=True)
        
        return all_user_scores

    def get_product_insights(self, product_name):
        """Get insights about users who already bought this product"""
        # Get users who bought this product
        buyers = self.transactions_df[self.transactions_df['TRANSACTION_BILLER'] == product_name]
        
        if len(buyers) == 0:
            return None
        
        # Total users who bought
        total_buyers = len(buyers['CIF'].unique())
        
        # Merge with user data
        buyer_details = pd.merge(buyers, self.users_df, on='CIF', how='left')
        
        insights = {}
        insights['total_buyers'] = total_buyers
        
        # Age distribution
        buyer_details['AGE_GROUP'] = pd.cut(buyer_details['AGE'], 
                                           bins=[0, 20, 25, 30, 35, 40, 50, 100],
                                           labels=['<20', '20-25', '25-30', '30-35', '35-40', '40-50', '>50'])
        age_dist = buyer_details['AGE_GROUP'].value_counts().head(3)
        insights['age_distribution'] = age_dist
        
        # Top 3 occupations
        occ_dist = buyer_details['OCCUPATION'].value_counts().head(3)
        insights['occupation_distribution'] = occ_dist
        
        # Gender distribution
        gender_dist = buyer_details['GENDER'].value_counts()
        insights['gender_distribution'] = gender_dist
        
        # Balance distribution
        balance_dist = buyer_details['RANGE_SALDO'].value_counts().head(3)
        insights['balance_distribution'] = balance_dist
        
        # Related products (products bought together)
        buyer_cifs = buyers['CIF'].unique()
        other_transactions = self.transactions_df[
            (self.transactions_df['CIF'].isin(buyer_cifs)) & 
            (self.transactions_df['TRANSACTION_BILLER'] != product_name)
        ]
        
        related_products = other_transactions['TRANSACTION_BILLER'].value_counts().head(3)
        insights['related_products'] = related_products
        
        return insights

    def get_score_distribution(self, user_scores, user_type='ALL'):
        """Get score distribution for specific user type"""
        if not user_scores:
            return pd.DataFrame()
        
        df = pd.DataFrame(user_scores)
        
        if user_type == 'ACTIVE':
            df = df[df['USER_STATUS'] == 'ACTIVE']
        elif user_type == 'COLD':
            df = df[df['USER_STATUS'] == 'COLD']
        
        # Define score ranges
        bins = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
        labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-90%', '90-100%']
        
        if len(df) > 0:
            df['SCORE_RANGE'] = pd.cut(df['FINAL_SCORE'], bins=bins, labels=labels)
            distribution = df['SCORE_RANGE'].value_counts().reindex(labels).fillna(0)
        else:
            distribution = pd.Series([0]*len(labels), index=labels)
        
        return distribution

# ==================== 5. STREAMLIT DASHBOARD ====================
def main():
    st.title("🏦 BANK BTN Product Recommendation System")
    st.markdown("---")
    
    # Initialize session state
    if 'recommender' not in st.session_state:
        st.session_state.recommender = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'selected_product' not in st.session_state:
        st.session_state.selected_product = None
    if 'insights' not in st.session_state:
        st.session_state.insights = None
    if 'sort_config' not in st.session_state:
        st.session_state.sort_config = {'column': 'FINAL_SCORE', 'ascending': False}
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Configuration")
        
        # File upload or path input
        st.subheader("Data Source")
        data_source = st.radio("Choose data source:", 
                              ["Use sample data", "Upload CSV files", "Use file paths"])
        
        users_df = None
        transactions_df = None
        
        if data_source == "Upload CSV files":
            users_file = st.file_uploader("Upload users.csv", type=['csv'])
            transactions_file = st.file_uploader("Upload transactions.csv", type=['csv'])
            
            if users_file and transactions_file:
                users_df = pd.read_csv(users_file)
                transactions_df = pd.read_csv(transactions_file)
                
        elif data_source == "Use file paths":
            users_path = st.text_input("Path to users.csv", "users.csv")
            transactions_path = st.text_input("Path to transactions.csv", "transactions.csv")
            
            if os.path.exists(users_path) and os.path.exists(transactions_path):
                users_df = pd.read_csv(users_path)
                transactions_df = pd.read_csv(transactions_path)
            else:
                st.error("File not found. Please check the paths.")
                
        else:  # Use sample data
            # Try to load from current directory
            try:
                users_df = pd.read_csv("users.csv")
                transactions_df = pd.read_csv("transactions.csv")
            except:
                st.warning("Sample files not found. Please upload or provide paths.")
        
        if users_df is not None and transactions_df is not None:
            st.success(f"✅ Data loaded: {len(users_df)} users, {len(transactions_df)} transactions")
            
            # Model weights
            st.subheader("Model Weights")
            cf_weight = st.slider("Collaborative Filtering Weight", 0.0, 1.0, 0.3, 0.1)
            cbf_weight = st.slider("Content-Based Filtering Weight", 0.0, 1.0, 0.7, 0.1)
            
            # Train button
            if st.button("🚀 Train Model", type="primary", use_container_width=True):
                with st.spinner("⏳ **Training model butuh 1-3 menit ya!** Lagi belajar dari data..."):
                    st.session_state.recommender = ProductRecommender(cf_weight=cf_weight, cbf_weight=cbf_weight)
                    st.session_state.recommender.fit(users_df, transactions_df)
                    st.success("🎉 **Model trained successfully!**")
        
        st.markdown("---")
        st.markdown("### 🎯 About")
        st.markdown("""
        This system recommends potential customers for specific banking products using:
        - **Item-Based Collaborative Filtering**
        - **Content-Based Filtering**
        - **Hybrid Recommendation**
        
        ⏱️ **Note:** Analysis might take 3-10 minutes for large datasets
        """)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("📈 Product Analysis")
        
        if st.session_state.recommender and st.session_state.recommender.is_trained:
            # Get unique products for dropdown
            unique_products = st.session_state.recommender.transactions_df['TRANSACTION_BILLER'].unique().tolist()
            unique_products.sort()
            
            # Product selection
            selected_product = st.selectbox(
                "Select a product to analyze:",
                unique_products,
                index=0 if len(unique_products) > 0 else None,
                key="product_select"
            )
            
            # Analysis button
            if st.button("🔍 Start Analysis", type="primary", use_container_width=True):
                st.session_state.selected_product = selected_product
                
                # Tampilkan pesan lucu sebelum mulai
                analysis_messages = [
                    "⏳ **Siap-siap ya!** Analisis bakal butuh **3-8 menit** nih!",
                    "🕐 **Tunggu sebentar!** Proses ini butuh **4-10 menit**, worth it kok!",
                    "⏰ **Mohon bersabar!** Loading sekitar **3-7 menit** ya!",
                    "⌛ **Sedang diproses!** Butuh waktu **5-9 menint** nih!"
                ]
                
                st.info(random.choice(analysis_messages))
                
                with st.spinner("🔍 **Analyzing product...**"):
                    # Get recommendations
                    st.session_state.results = st.session_state.recommender.get_all_users_for_product(
                        selected_product, min_score=0.01
                    )
                    
                    # Get product insights
                    st.session_state.insights = st.session_state.recommender.get_product_insights(selected_product)
        
    with col2:
        st.header("📥 Export")
        if st.session_state.results:
            # Create Excel file
            df_results = pd.DataFrame(st.session_state.results)
            
            # Prepare Excel with multiple sheets
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # All users
                df_all = df_results[['CIF', 'FINAL_SCORE']].copy()
                df_all.to_excel(writer, sheet_name='All Users', index=False)
                
                # Potential buyers
                df_potential = df_results[df_results['USER_STATUS'] == 'ACTIVE'][['CIF', 'FINAL_SCORE']].copy()
                df_potential.to_excel(writer, sheet_name='Potential Buyers', index=False)
                
                # Cold users (CBF only)
                cold_users = st.session_state.recommender.users_df[
                    st.session_state.recommender.users_df['IS_COLD'] == True
                ].copy()
                
                cold_scores = []
                for _, user in cold_users.iterrows():
                    cbf_score = st.session_state.recommender.cbf_model.get_cbf_score_fast(
                        user['CIF'], st.session_state.selected_product
                    )
                    if cbf_score >= 0.01:
                        cold_scores.append({
                            'CIF': user['CIF'],
                            'FINAL_SCORE': round(cbf_score, 4)
                        })
                
                df_cold = pd.DataFrame(cold_scores)
                df_cold.to_excel(writer, sheet_name='Cold User Potential', index=False)
            
            # Download button
            st.download_button(
                label="📊 Download Excel Report",
                data=output.getvalue(),
                file_name=f"recommendations_{st.session_state.selected_product}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    # Display results if available
    if st.session_state.results and st.session_state.insights:
        st.markdown("---")
        
        # Product Insights Section
        st.subheader(f"📊 Insights for: {st.session_state.selected_product}")
        
        insights = st.session_state.insights
        
        # Row 1: Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Total buyers
            st.metric("Users who already bought", 
                     f"{insights['total_buyers']:,}",
                     "Actual buyers")
        
        with col2:
            # Most common age group
            if 'age_distribution' in insights and len(insights['age_distribution']) > 0:
                st.metric("Most Common Age Group", 
                         insights['age_distribution'].index[0],
                         f"{insights['age_distribution'].iloc[0]:,} users")
        
        with col3:
            # Gender distribution
            if 'gender_distribution' in insights and len(insights['gender_distribution']) > 0:
                total_gender = insights['gender_distribution'].sum()
                male_pct = insights['gender_distribution'].get('M', 0) / total_gender * 100
                female_pct = insights['gender_distribution'].get('F', 0) / total_gender * 100
                
                if male_pct >= female_pct:
                    st.metric("Dominant Gender", "Male", f"{male_pct:.1f}%")
                else:
                    st.metric("Dominant Gender", "Female", f"{female_pct:.1f}%")
        
        with col4:
            # Most common balance
            if 'balance_distribution' in insights and len(insights['balance_distribution']) > 0:
                st.metric("Most Common Balance", 
                         insights['balance_distribution'].index[0],
                         f"{insights['balance_distribution'].iloc[0]:,} users")
        
        # Row 2: Top 3 Occupations
        st.subheader("👥 Top 3 Occupations of Buyers")
        
        if 'occupation_distribution' in insights and len(insights['occupation_distribution']) > 0:
            occ_cols = st.columns(3)
            
            for idx, (occupation, count) in enumerate(insights['occupation_distribution'].items()):
                if idx < 3:
                    with occ_cols[idx]:
                        # Calculate percentage
                        total_buyers = insights['total_buyers']
                        percentage = (count / total_buyers * 100) if total_buyers > 0 else 0
                        
                        st.metric(
                            f"{idx+1}. {occupation}",
                            f"{count:,}",
                            f"{percentage:.1f}%"
                        )
        
        # Products often bought together
        st.subheader("🛒 Products Often Bought Together")
        if 'related_products' in insights and len(insights['related_products']) > 0:
            rel_cols = st.columns(min(3, len(insights['related_products'])))
            for idx, (product, count) in enumerate(insights['related_products'].items()):
                if idx < 3:
                    with rel_cols[idx]:
                        # Calculate percentage of buyers who also bought this
                        percentage = (count / insights['total_buyers'] * 100) if insights['total_buyers'] > 0 else 0
                        st.metric(product, f"{count:,}", f"{percentage:.1f}%")
        
        st.markdown("---")
        
        # Score Distribution Charts
        st.subheader("📈 Customer Potential Distribution")
        
        # Get distributions for all user types
        dist_all = st.session_state.recommender.get_score_distribution(
            st.session_state.results, 'ALL'
        )
        dist_active = st.session_state.recommender.get_score_distribution(
            st.session_state.results, 'ACTIVE'
        )
        dist_cold = st.session_state.recommender.get_score_distribution(
            st.session_state.results, 'COLD'
        )
        
        # Create charts
        tab1, tab2, tab3 = st.tabs(["All Users", "Potential Customers (Active)", "Cold Users"])
        
        with tab1:
            if len(dist_all) > 0 and dist_all.sum() > 0:
                fig = px.bar(
                    x=dist_all.index,
                    y=dist_all.values,
                    title="All Users Distribution",
                    labels={'x': 'Score Range', 'y': 'Number of Users'},
                    color=dist_all.values,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display numbers
                st.subheader("📊 Count by Score Range")
                cols = st.columns(len(dist_all))
                for idx, (score_range, count) in enumerate(dist_all.items()):
                    with cols[idx]:
                        st.metric(score_range, f"{int(count):,}")
            else:
                st.info("No data available for All Users")
        
        with tab2:
            if len(dist_active) > 0 and dist_active.sum() > 0:
                fig = px.bar(
                    x=dist_active.index,
                    y=dist_active.values,
                    title="Potential Customers (Active Users) Distribution",
                    labels={'x': 'Score Range', 'y': 'Number of Users'},
                    color=dist_active.values,
                    color_continuous_scale='Greens'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display numbers
                st.subheader("📊 Count by Score Range")
                cols = st.columns(len(dist_active))
                for idx, (score_range, count) in enumerate(dist_active.items()):
                    with cols[idx]:
                        st.metric(score_range, f"{int(count):,}")
            else:
                st.info("No potential customers found")
        
        with tab3:
            if len(dist_cold) > 0 and dist_cold.sum() > 0:
                fig = px.bar(
                    x=dist_cold.index,
                    y=dist_cold.values,
                    title="Cold Users Distribution",
                    labels={'x': 'Score Range', 'y': 'Number of Users'},
                    color=dist_cold.values,
                    color_continuous_scale='Reds'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display numbers
                st.subheader("📊 Count by Score Range")
                cols = st.columns(len(dist_cold))
                for idx, (score_range, count) in enumerate(dist_cold.items()):
                    with cols[idx]:
                        st.metric(score_range, f"{int(count):,}")
            else:
                st.info("No cold users found")
        
        st.markdown("---")
        
        # Detailed Data Table
        st.subheader("📋 Detailed Customer Data")
        
        df_display = pd.DataFrame(st.session_state.results)
        
        # Add score range for filtering
        df_display['SCORE_RANGE'] = pd.cut(
            df_display['FINAL_SCORE'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
            labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-90%', '90-100%']
        )
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            score_filter = st.multiselect(
                "Filter by Score Range:",
                options=df_display['SCORE_RANGE'].unique(),
                default=df_display['SCORE_RANGE'].unique()
            )
        
        with col2:
            status_filter = st.multiselect(
                "Filter by User Status:",
                options=df_display['USER_STATUS'].unique(),
                default=df_display['USER_STATUS'].unique()
            )
        
        with col3:
            min_score = st.slider("Minimum Score:", 0.0, 1.0, 0.01, 0.01)
        
        # Apply filters
        filtered_df = df_display[
            (df_display['SCORE_RANGE'].isin(score_filter)) &
            (df_display['USER_STATUS'].isin(status_filter)) &
            (df_display['FINAL_SCORE'] >= min_score)
        ].copy()
        
        # Sorting functionality yang benar
        sort_col = st.selectbox(
            "Sort by column:",
            ['FINAL_SCORE', 'AGE', 'GENDER', 'OCCUPATION', 'RANGE_SALDO', 'USER_STATUS']
        )
        
        sort_order = st.radio(
            "Sort order:",
            ["Descending (High to Low)", "Ascending (Low to High)"],
            horizontal=True
        )
        
        ascending = sort_order == "Ascending (Low to High)"
        
        # Apply sorting
        if sort_col == 'FINAL_SCORE':
            filtered_df = filtered_df.sort_values('FINAL_SCORE', ascending=ascending)
        elif sort_col == 'AGE':
            filtered_df = filtered_df.sort_values('AGE', ascending=ascending)
        elif sort_col == 'GENDER':
            filtered_df = filtered_df.sort_values('GENDER', ascending=ascending)
        elif sort_col == 'OCCUPATION':
            filtered_df = filtered_df.sort_values('OCCUPATION', ascending=ascending)
        elif sort_col == 'RANGE_SALDO':
            # Urutkan berdasarkan kategori balance yang sudah ada
            balance_order = ['1-5jt', '5-10jt', '10-25jt', '25-50jt', '50-100jt', '>100jt']
            if ascending:
                balance_order = balance_order
            else:
                balance_order = list(reversed(balance_order))
            
            filtered_df['RANGE_SALDO'] = pd.Categorical(
                filtered_df['RANGE_SALDO'], 
                categories=balance_order, 
                ordered=True
            )
            filtered_df = filtered_df.sort_values('RANGE_SALDO', ascending=ascending)
        elif sort_col == 'USER_STATUS':
            filtered_df = filtered_df.sort_values('USER_STATUS', ascending=ascending)
        
        # Display table dengan interaktif sorting dari Streamlit
        st.dataframe(
            filtered_df[['CIF', 'FINAL_SCORE', 'SCORE_RANGE', 'USER_STATUS', 'AGE', 'GENDER', 'OCCUPATION', 'RANGE_SALDO']]
            .head(100),
            use_container_width=True,
            column_config={
                "FINAL_SCORE": st.column_config.NumberColumn(
                    "Score",
                    format="%.4f",
                    help="Recommendation score (0-1)"
                ),
                "AGE": st.column_config.NumberColumn(
                    "Age",
                    help="Customer age"
                ),
                "SCORE_RANGE": st.column_config.TextColumn(
                    "Score Range",
                    help="Score category"
                ),
                "USER_STATUS": st.column_config.TextColumn(
                    "Status",
                    help="Active or Cold user"
                )
            }
        )
        
        st.caption(f"Showing {min(len(filtered_df), 100)} of {len(filtered_df)} customers")
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Potential Customers", f"{len(df_display):,}")
        
        with col2:
            active_count = len(df_display[df_display['USER_STATUS'] == 'ACTIVE'])
            st.metric("Active Users", f"{active_count:,}")
        
        with col3:
            cold_count = len(df_display[df_display['USER_STATUS'] == 'COLD'])
            st.metric("Cold Users", f"{cold_count:,}")
        
        with col4:
            avg_score = df_display['FINAL_SCORE'].mean()
            st.metric("Average Score", f"{avg_score:.3f}")
    
    elif st.session_state.recommender and not st.session_state.results:
        st.info("👆 Select a product and click 'Start Analysis' to begin")
    
    elif not st.session_state.recommender:
        st.info("👈 Please load data and train the model first in the sidebar")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        🏦 BANK BTN Product Recommendation System | Built with Streamlit | ⏱️ Analysis time: 3-10 minutes
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    main()
