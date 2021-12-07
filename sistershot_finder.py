import os,shutil,glob,math,imagehash
import numpy as np
import pandas as pd
import urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
import networkx as nx

def candidate_collective(df_original):
    df = pd.read_sql("""
    SELECT 
        mp.user_id, 
        mp.photo_id,
        'https://cdn.eyeem.com/thumb/' || p.file_id || '/600/600' as url,
        '=IMAGE("' || 'https://cdn.eyeem.com/thumb/' || p.file_id || '/400/400' || '",1)' as image,
        ep.created_at::date
    FROM eyescream.market_photo mp
    JOIN eyescream.eyeem_photo ep on ep.id = mp.photo_id
    JOIN inspectorprod.photos p on p.photo_id = mp.photo_id
    WHERE 1=1 
    and p.file_id is not null
    and mp.status != 5
    and mp.user_id in {0}""".format(str(tuple(df_original['user_id']))), con=con)
    return df.drop_duplicates(subset=['photo_id'], keep='last')

def collective_caption(df_original):
    con = create_conn(config=config)
    df = pd.read_sql("""
    WITH m as (
      SELECT
          mp.photo_id,
          caption,
          ae.score AS aesthetics_score,
          mp.created_at::date as created_at,
          ROW_NUMBER() OVER (PARTITION BY cap.photo_id ORDER BY cap.score desc) as rk
      FROM eyescream.market_photo mp
      JOIN eyescream.eyeem_photo ep on ep.id = mp.photo_id
      JOIN koala.captions_latest cap on cap.photo_id = mp.photo_id
      LEFT JOIN koala.aesthetics_latest ae ON ae.photo_id = mp.photo_id
      WHERE 1=1 
      and mp.status != 5
      and mp.user_id in {0}
      and ep.created_at::date > '2010-01-01'
      and cap.caption != '' )

    SELECT m.photo_id, lower(m.caption) as caption, m.aesthetics_score
    from m
    where 1=1 AND rk = 1""".format(str(tuple(df_original['user_id']))), con=con)
    con.close()
    df = pd.merge(df_original, df, how='inner', on='photo_id')
    return df.drop_duplicates(subset=['photo_id'], keep='last')

def collective_kw(df_original):
    con = create_conn(config=config)
    df = pd.read_sql("""
    select mp.photo_id, listagg(keywords, ',') within group (order by keywords) as keywords
    FROM eyescream.market_photo mp
    JOIN eyescream.eyeem_photo ep on ep.id = mp.photo_id
    JOIN koala.concepts_latest cv on cv.photo_id = mp.photo_id
    where mp.user_id in {0}
    and mp.status != 5
    group by 1 """.format(str(tuple(df_original['user_id']))), con=con)
    con.close()
    df = pd.merge(df_original, df, how='inner', on='photo_id')
    return df.drop_duplicates(subset=['photo_id'], keep='last')

def re_clean(df):
    df = df[(~df['keywords'].isnull()) & (~df['caption'].isnull())]
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['wn'] = df['created_at'].dt.strftime('%Y-%U')
    df['mth'] = df['created_at'].dt.strftime('%Y-%m')
    df['yr'] = df['created_at'].dt.strftime('%Y')
    df = df[['photo_id', 'user_id', 'url', 'image', 'created_at',
             'keywords', 'caption', 'aesthetics_score', 'wn', 'mth', 'yr']]

    return df

def single_hashing_cal(x, r):
    img = Image.open(r)
    dhash,whash,phash = imagehash.dhash_vertical(img),imagehash.whash(img),imagehash.phash(img)
    df = pd.DataFrame({'photo_id':[x], 'hash':[[dhash,whash,phash]]})
    return df

def gettodown_one(ud, url, photoid):
    try:
        photoid = str(photoid)
        root = './'+ ud +'/temp_image/' + photoid + '.jpg'
        urllib.request.urlretrieve(url, root)
        df = single_hashing_cal(photoid, root)
        os.remove(root)
        return df
    except:
        pass

def tfidf_machine_kw(df):
    dataset = [str(x).replace(',', ' ') for x in df['keywords'].unique()]
    photo_id = df['photo_id'].tolist()
    try:
        vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
        X_tfidf = vectorizer.fit_transform(dataset)
        return X_tfidf
    except:
        vectorizer = TfidfVectorizer(stop_words='english')
        X_tfidf = vectorizer.fit_transform(dataset)
        return X_tfidf

def tfidf_machine_cap(df):
    photo_id = df['photo_id'].tolist()

    try:
        vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
        X_tfidf = vectorizer.fit_transform(df['caption'].tolist())
        return X_tfidf
    except:
        vectorizer = TfidfVectorizer(stop_words='english')
        X_tfidf = vectorizer.fit_transform(df['caption'].tolist())
        return X_tfidf

def awesome_cossim_top(A, B, ntop, lower_bound):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape

    idx_dtype = np.int32

    nnz_max = M * M

    indptr = np.zeros(M + 1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        M,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data, indices, indptr), shape=(M, N))

def get_matches_df(sparse_matrix, name_vector, top=100):
    non_zeros = sparse_matrix.nonzero()
    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]
    left_side = np.empty([len(sparse_matrix.data)], dtype=object)
    right_side = np.empty([len(sparse_matrix.data)], dtype=object)
    similairity = np.zeros(len(sparse_matrix.data))

    for index in range(0, len(sparse_matrix.data)):
        left_side[index] = name_vector[sparserows[index]]
        right_side[index] = name_vector[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]

    return pd.DataFrame({'a': left_side,
                         'b': right_side,
                         'score': similairity})

def kw_tfidf(df):
    data_matrix = tfidf_machine_kw(df)
    matches = awesome_cossim_top(data_matrix, data_matrix.transpose(), 300, 0.3)
    matches = get_matches_df(matches, df.photo_id.tolist(), top=297380)
    matches = matches[matches['a'] != matches['b']]
    return matches

def cap_tfidf(df, kw):
    data_matrix = tfidf_machine_cap(df)
    matches = awesome_cossim_top(data_matrix, data_matrix.transpose(), 300, 0.1)
    matches = get_matches_df(matches, df.photo_id.tolist(), top=297380)
    matches = matches[matches['a'] != matches['b']]
    matches.columns = ['a', 'b', 'caption_score']

    kw = pd.merge(kw, matches, how='inner', on=['a', 'b'])
    kw['compound_score'] = kw['score'] + kw['caption_score']
    kw = kw[(kw['compound_score'] >= 0.7) & (kw['a'] != kw['b'])]
    kw = kw.apply(pd.to_numeric, errors='ignore')
    return kw

def processkw(df):
    kw_df = kw_tfidf(df)
    return cap_tfidf(df, kw_df)

def small_folder_tfidf(original):
    original = original.sort_values('photo_id', ascending=True)
    temp = []
    base = math.ceil(original.shape[0] / 4)

    small_df = original.iloc[:base]
    temp.append(processkw(small_df))
    small_df = original.iloc[base:base * 2]
    temp.append(processkw(small_df))
    small_df = original.iloc[base * 2:base * 3]
    temp.append(processkw(small_df))
    small_df = original.iloc[base * 3:]
    temp.append(processkw(small_df))

    return pd.concat(temp)

def over_7000(daw):
    df = pd.DataFrame()

    if daw.shape[0] <= 6000:
        df = processkw(daw)
    else:
        if daw.shape[0] <= 22000:
            df = small_folder_tfidf(daw).drop_duplicates(subset=['a', 'b'], keep='last')
        for www in [daw[daw['wn'] == m] for m in daw['wn'].unique()]:
            if www.shape[0] > 6000:
                for wwwjm in [www[www['created_at'] == m] for m in www['created_at'].unique()]:
                    if wwwjm.shape[0] > 6000:
                        df = pd.concat([df, small_folder_tfidf(wwwjm)]).drop_duplicates(subset=['a', 'b'], keep='last')
                    else:
                        df = pd.concat([df, processkw(wwwjm)]).drop_duplicates(subset=['a', 'b'], keep='last')
            else:
                df = pd.concat([df, processkw(www)]).drop_duplicates(subset=['a', 'b'], keep='last')

    return df.drop_duplicates(subset=['a', 'b'], keep='last')

def basic_clean(route):
    tempimage = './' + route + '/temp_image/'
    temp_folder = './' + route + '/kw_temp/'

    if os.path.exists(tempimage):
        shutil.rmtree(tempimage)
    os.makedirs(tempimage)
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    os.makedirs(temp_folder)

    if not os.path.exists('./' + route + '/csv/'):
        os.makedirs('./' + route + '/csv/')

    if not os.path.exists('./' + route + '/gsheet/'):
        os.makedirs('./' + route + '/gsheet/')

def hashing_compare(imageone, imagetwo, original, target, route):
    dhash_num = original[0] - target[0]
    df = pd.DataFrame()
    if dhash_num <= 20:
        whash_num = original[1] - target[1]
        if whash_num <= 20:
            phash_num = original[2] - target[2]
            if phash_num <= 20:
                a = pd.DataFrame({'a': [imageone], 'b': [imagetwo],
                                  'dvhas': [dhash_num], 'whash': [whash_num],
                                  'phash': [phash_num]})
                b = pd.DataFrame({'a': [imagetwo], 'b': [imageone],
                                  'dvhas': [dhash_num], 'whash': [whash_num],
                                  'phash': [phash_num]})

                filename = route + str(imageone) + '_' + str(imagetwo) + '.csv'
                pd.concat([a, b]).to_csv(filename, index=False)

def create_end(route):
    end = [pd.DataFrame()]
    for f in glob.glob(route + '/*.csv'):
        try:
            end.append(pd.read_csv(f))
        except:
            pass
    return pd.concat(end).drop_duplicates(subset=['a','b'],keep='last')

def output_df(fb, end):
    end = end.apply(pd.to_numeric, errors='ignore')
    fb = fb.apply(pd.to_numeric, errors='ignore')
    ek = pd.merge(fb, end, on=['a', 'b'], how='inner').drop_duplicates(subset=['a', 'b'], keep='last')
    ek['compound_hash'] = ek['dvhas'] + ek['whash'] + ek['phash']

    e1 = ek[(ek['score'] > 0.8)]
    e5 = ek[(ek['compound_score'] > 1.5) & (ek['score'] > 0.7) & (ek['compound_hash'] <= 45)]

    ek = ek[(ek['compound_hash'] <= 45) & (ek['compound_score'] <= 45)]
    e2 = ek[(ek['dvhas'] < 16) & (ek['whash'] < 16) & (ek['phash'] <= 8)]
    e3 = ek[(ek['dvhas'] <= 8) & (ek['whash'] < 16) & (ek['phash'] < 16)]
    e4 = ek[(ek['dvhas'] < 16) & (ek['whash'] <= 8) & (ek['phash'] < 16)]

    # put together
    df = pd.concat([e1, e3, e4, e2, e5]).drop_duplicates(subset=['a', 'b'], keep='last')
    return df

def graph_to_csv(df, original, root_route):
    G = nx.Graph()
    G.add_edges_from(list(df[['a', 'b']].itertuples(index=False, name=None)))
    group_num = nx.number_connected_components(G)
    group = list(nx.connected_components(G))
    if group_num > 0:
        large_output(group_num, group, original, root_route)

def large_output(group_num, g, df, route):
    dubp = set()
    df['group'] = 0
    for positi in range(0, group_num):
        df['group'] = np.where(df['photo_id'].isin(g[positi]), positi + 1, df['group'])
    df = df[df['group'] != 0]
    df = df.assign(groups=lambda x: x.group * 1 / 50)
    df['groups'] = df['groups'].apply(lambda x: math.ceil(x))

    for big_group in df.groups.unique():
        pc = df[df.groups == big_group].copy()  # 50 packs per group
        ex = './' + route + '/gsheet/' + wk + '_' + str(big_group) + '.xlsx'

        writer = pd.ExcelWriter(ex, engine='xlsxwriter')
        for grp in pc.group.unique():
            filename = 'pack_' + str(grp)
            dfpc = pc[pc['group'] == grp].sort_values('aesthetics_score', ascending=False)
            best_shot = dfpc['photo_id'].iloc[0]

            bestone = dfpc[dfpc['photo_id'] == best_shot][['photo_id', 'image']].reset_index(drop=True)
            bestone.columns = ['best_shot', 'thumbnail_bestshot']
            duplicate = dfpc[dfpc['photo_id'] != best_shot][['photo_id', 'image']].reset_index(drop=True)
            duplicate.columns = ['replica', 'thumbnail_replica']
            dubp = dubp | set(duplicate.replica)
            msa = pd.concat([bestone, duplicate], axis=1)
            msa.to_excel(writer, index=False, sheet_name=filename)

            worksheet = writer.sheets[filename]  # Access the Worksheet
            worksheet.set_column(1, 1, 45)
            worksheet.set_column(3, 3, 45)
            height = msa.shape[0]
            for i in range(1, height + 1):
                worksheet.set_row(i, 120)
            writer.sheets[filename] = worksheet
        writer.save()

    if len(dubp) > 0:
        csv_route = './' + route + '/csv/' + route + '_sisterfinder.csv'
        pd.DataFrame({'photo_id': list(dubp)}).to_csv(csv_route, index=False, header=False)
        output.to_csv('./' + route + '/score.csv', index=False, encoding='utf-8')


dd = None # a list of user_id in DataFrame
dd = candidate_collective(dd)
dd = collective_caption(dd)
dd = collective_kw(dd)
dd = re_clean(dd)

# per user
for userid_int in dd.user_id.unique():
    df = dd[dd['user_id'] == userid_int]
    folder_name = str(userid_int)
    # compare kw and caption
    fb = over_7000(df)
    photo_list = list(set(fb.a) | set(fb.b))
    edc = dd[dd['photo_id'].isin(photo_list)]
    basic_clean(folder_name)

    # download & hashing
    f = [gettodown_one(folder_name, x2.url, x2.photo_id) for x1, x2 in edc.iterrows()]
    f = [x for x in f if x is not None]

    if len(f) > 1:
        hx = pd.concat(f).apply(pd.to_numeric, errors='ignore')
        fb = fb[(fb['a'].isin(set(hx.photo_id))) & (fb['b'].isin(set(hx.photo_id)))]
        candidate = set([tuple(r) for r in fb[['a', 'b']].to_numpy()])

        # compare hashing score
        if len(candidate) > 0:
            temp_folder = './' + folder_name + '/kw_temp/'
            yy = pd.DataFrame(candidate)
            yy.columns = ['a', 'b']
            for name, group in yy.groupby('a'):
                ori = hx[hx['photo_id'] == name]['hash'].iloc[0]
                for target in group.b.tolist():
                    ta = hx[hx['photo_id'] == target]['hash'].iloc[0]
                    file_name = temp_folder + str(name) + '_' + str(target) + '.csv'
                    reverse_name = temp_folder + str(target) + '_' + str(name) + '.csv'
                    if (not os.path.exists(file_name)) & (not os.path.exists(reverse_name)):
                        hashing_compare(name, target, ori, ta, temp_folder)

            hx, yy, candidate, edc = None, None, None, None
            end = create_end(temp_folder)
            if end.shape[0] > 0:
                output = output_df(fb, end)
                if output.shape[0] > 0:
                    graph_to_csv(output, dd, folder_name)

    shutil.rmtree(folder_name)