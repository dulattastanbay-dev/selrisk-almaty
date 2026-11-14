/* Numeric tree inference. RF casts inputs to float32; HistGB uses float64.
   JSON contains fitted trees, not a surrogate or a logistic fallback. */
(function(root){
  "use strict";
  const RF="random_forest_inventory_background";
  const GB="hist_gradient_boosting_assumed_negative_trigger";
  function validate(model){
    if(!model||model.version!=="0.3"||![RF,GB].includes(model.model_type)||
       !Array.isArray(model.features)||!model.features.length||
       new Set(model.features).size!==model.features.length||
       !Array.isArray(model.trees)||!model.trees.length||
       !["feature_min","feature_max"].every(k=>Array.isArray(model[k])&&
         model[k].length===model.features.length&&model[k].every(Number.isFinite))||
       model.feature_min.some((x,i)=>x>model.feature_max[i]))
      throw Error("Некорректный формат обученной модели");
    if(model.model_type===GB&&(!Number.isFinite(model.baseline)||
       model.leaf_values_include_learning_rate!==true))throw Error("Некорректный бустинг");
    for(const tree of model.trees){
      if(!Array.isArray(tree)||!tree.length)throw Error("Пустое дерево");
      tree.forEach((n,i)=>{
        if(!Array.isArray(n)||n.length!==5||!n.every(Number.isFinite))throw Error("Некорректный узел");
        const [left,right,feature,threshold,value]=n;
        if(left===-1){
          if(right!==-1||feature!==-1||(model.model_type===RF&&(value<0||value>1)))
            throw Error("Некорректный лист");
        }else if(![left,right,feature].every(Number.isInteger)||left<=i||right<=i||
                 left>=tree.length||right>=tree.length||feature<0||feature>=model.features.length)
          throw Error("Некорректные ветви дерева");
      });
    }
    return model;
  }
  function leaf(tree,values){
    let i=0;
    while(tree[i][0]!==-1){const n=tree[i];i=values[n[2]]<=n[3]?n[0]:n[1];}
    return tree[i][4];
  }
  function predict(model,row){
    const x=model.features.map(k=>row[k]);
    if(x.some(v=>typeof v!=="number"||!Number.isFinite(v)))return {value:null,status:"missing_features"};
    if(x.some((v,i)=>v<model.feature_min[i]-1e-7||v>model.feature_max[i]+1e-7))
      return {value:null,status:"out_of_training_domain"};
    if(model.model_type===RF){
      const values=x.map(Math.fround);
      return {value:model.trees.reduce((s,t)=>s+leaf(t,values),0)/model.trees.length,status:"available"};
    }
    if(model.model_type!==GB)throw Error("Ожидались случайный лес или бустинг");
    const raw=model.trees.reduce((s,t)=>s+leaf(t,x),model.baseline);
    return {value:raw>=0?1/(1+Math.exp(-raw)):Math.exp(raw)/(1+Math.exp(raw)),status:"available"};
  }
  function fuse(spatial,trigger){
    if(![spatial,trigger].every(p=>typeof p==="number"&&Number.isFinite(p)&&p>=0&&p<=1))
      throw Error("Нет расчёта обоих блоков модели");
    return spatial*trigger;
  }
  const api={validate,predict,fuse};root.SelRiskTrees=api;
  if(typeof module!=="undefined")module.exports=api;
})(typeof globalThis!=="undefined"?globalThis:this);
