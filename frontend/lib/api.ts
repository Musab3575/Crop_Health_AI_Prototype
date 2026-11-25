export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000/api/v1";

export type Alternative = { class_id:number; class_name:string; crop_name:string; disease_name:string; confidence:number };
export type Recommendation = { crop:string; status:string; description:string; symptoms:string[]; possible_factors:string[]; prevention:string[]; management:string[]; monitoring:string[]; seek_expert_when:string[] };
export type Prediction = { prediction_id:string; predicted_class_id:number; predicted_class_name:string; crop_name:string; disease_name:string; is_healthy:boolean; confidence:number; top_predictions:Alternative[]; recommendation:Recommendation; explanation:string; low_confidence_warning?:string; domain_status:'supported_prediction'|'uncertain'|'outside_domain'; domain_warning?:string; plant_like_score:number; image_info:Record<string,unknown>; timestamp:string };
export type HistoryPage = { items:Prediction[]; total:number; page:number; page_size:number; pages:number };

async function request<T>(path:string, init?:RequestInit):Promise<T>{
  const controller=new AbortController(); const timer=setTimeout(()=>controller.abort(),30000);
  try {
    let response:Response;
    try { response=await fetch(`${API_BASE}${path}`,{...init,signal:controller.signal}); }
    catch(error) {
      if(error instanceof DOMException && error.name==="AbortError") throw new Error("The API request timed out. Check that the backend is running and try again.");
      if(error instanceof TypeError) throw new Error(`Cannot reach the Crop Health AI backend at ${API_BASE}. Start the backend and refresh this page.`);
      throw error;
    }
    if(!response.ok){let message="Request failed"; try{const data=await response.json(); message=data.detail||message}catch{} throw new Error(message)} return await response.json();
  }
  finally {clearTimeout(timer)}
}
export async function predict(file:Blob, name="leaf-image.jpg") { const form=new FormData(); form.append("file",file,name); return request<Prediction>("/predict",{method:"POST",body:form}); }
export const getHealth=()=>request<{status:string;model_status:string;app_version:string;model_loaded:boolean}>('/health');
export const getStats=()=>request<any>('/stats');
export const getHistory=(page=1,pageSize=10,search="",status="",sort="newest")=>request<HistoryPage>(`/history?page=${page}&page_size=${pageSize}&search=${encodeURIComponent(search)}&status=${encodeURIComponent(status)}&sort=${sort}`);
export const getClasses=()=>request<any[]>('/classes');
export const deleteHistory=(id:string)=>request<any>(`/history/${id}`,{method:"DELETE"});
export const deleteAllHistory=()=>request<any>('/history',{method:"DELETE"});
