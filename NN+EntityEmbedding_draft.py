
# prepare features

# data types
def cat_or_num(df):
    df_types = {}
    types = df.fillna(np.nan).dtypes
    df_types['num_cols'] = types.index[types != 'object']
    df_types['cat_cols'] = df.columns - num_cols
    df_types.to_pickel('df_types')
    return df_types

# fillmissing 
def fill_cat(df):
    return df.fillna(None, inplace=True)
 
def fill_num(df):
    return df.fillna(0, inplace=True)
    
# label encoding
def label_encode(df):
    le = labelencode
    les = le.fit(df)
    les.transform(df)
    les.to_pickel('labelencoder')
    return df

# embedding size (cat data)
def emb_size(n_cat):
   # min(50, (n_cat//2)+1)
   return min(round(1.6*n_cat**0.56), 50)
  
def emb_shape(ser):
    n_cat = ser.nunique(dropna=False)
    size = emb_size(n_cat)
    return [n_cat, size]
    # zip?

def preprocessing(self, X):   
    cat_cols, num_cols = cat_or_num(X)    
    X[cat_cols] = fill_cat(X[cat_cols])
    X[num_cols] =  fill_num(X[num_cols])
    X[cat_cols] = label_encode(X[cat_cols]):
    X = pd.concat([X[num_cols], X[cat_cols]], axis=1)
    #num_idx = X.columns.get_loc(num_cols)
    #cat_idx = X.columns.get_loc(cat_cols)
    colnames = {}
    colnames['cat_cols'] = cat_cols
    colnames['num_cols'] = num_cols
    
    embedding_shape = {}
    for col in cat_cols:
        embedding_shape[col] = emb_shape(X[col])
     
    return X, colnames, embedding_shape
    
class NN_with_EntityEmbeddings(model):
    def __init__(data):
        super().__init__()
        self.epochs = 10        
        self.max_log_y = max(np.max(y_train),np.max(y_val))
        self.__build_keras_model()
        self.fit(X_train, y_train, X_val, y_val)        
    
        
    def __build_keras_model(self):
        # embedding layer
          
        input_model = []
        output_embeddings = []
        for col in num_cols:
            input_num = Input(shape=(1,))
            output_num = Dense(1)(input_num)
            
            input_model = input_model.append(input_num)
            output_embeddings = output_model.append(output_num)
            
        for col in cat_cols:
            input_cat = Input(shape=(1,))
            output_cat = Embedding(embedding_shape[col], name = col)(input_cat)
            output_cat = reshape(target_shape=embedding_shape[col][1])(out_put_cat)
            
            input_model = input_model.append(input_cat)
            output_embeddings = output_model.append(output_cat)
        
        output_model = Concatenate()(output_embeddings)
        # Layer 1
        output_model = Dense(1000, kernel_initializer='uniform')(output_model)        
        output_model = Activation('relu')(output_model)
        # Layer 2
        output_model = Dense(500, kernel_initializer='uniform')(output_model)       
        output_model = Activation('relu')(output_model)
        # Output layer
        output_model = Dense(1)(output_model)       
        output_model = Activation('sigmoid')(output_model)
        
        self.model = KerasModel(inputs=input_model, outputs=output_model)
        
        self.model.compile(loss='mean_absolute_error',optimizer='adam')
           
    def _val_for_fit(self, val):
        val = np.log(val) / self.max_log_y
        return val
       
    def _val_for_pred(self, val):
        return np.exp(val * self.max_log_y)
       
    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, self._val_for_fit(y_train),
                       validation_data=(X_val, self._val_for_fit(y_val)),
                       epochs=self.epochs, batch_size=64,
                       )        
        print("Result on validation data: ", self.evaluate(X_val, y_val))
        
    def guess(self, features):
        result = self.model.predict(features).flatten()
        return self._val_for_pred(result)
        
        
# client:
result = preprocessing(df)
NN_with_EntityEmbeddings(df, result)
        