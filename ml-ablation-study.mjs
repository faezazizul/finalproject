import * as ML from './ml-recommender.js';
function gen(n,seed=1){const r=ML.makeRandom(seed);
 const A=[{c:[0,1,2,3,4],o:[5,6,7,8,9]},{c:[10,11,12,13,14],o:[15,16,17,18,19]},{c:[20,21,22,23,24],o:[25,26,27,28,29]}];
 const out=[];for(let u=0;u<n;u++){const a=A[u%3];const w=[];
 for(const i of a.c)w.push({type:"strength",exerciseId:`ex_${i}`,sets:4,reps:8,weight:150+r()*100});
 for(const i of a.o)if(r()<0.5)w.push({type:"strength",exerciseId:`ex_${i}`,sets:3,reps:10,weight:60+r()*60});
 out.push({userId:`u${u}`,workouts:w});}return out;}
const m=ML.buildInteractionMatrix(gen(120));
console.log("negatives | precision@5 | recall@5 | lift vs popularity");
console.log("----------|-------------|----------|-------------------");
for (const ns of [0,1,2,4,8,16]) {
  const ev=ML.evaluate(m.interactions,{factors:8,epochs:120,k:5,negativeSamples:ns});
  const lift = ev.results.liftOverPopularity*100;
  console.log(`    ${String(ns).padStart(2)}    |   ${ev.results.model.precision.toFixed(4)}    |  ${ev.results.model.recall.toFixed(4)}  |  ${lift>=0?"+":""}${lift.toFixed(1)}%`);
}
