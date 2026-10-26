/* Timers stay on their own clock: server perf_counter for inference,
   browser performance.now for RTT/end-to-end. Never subtract wall clocks. */
'use strict';
const $ = id => document.getElementById(id);
const labels = ['with_mask','without_mask','mask_weared_incorrect','uncertain'];
const names = {with_mask:'Đeo đúng',without_mask:'Không đeo',mask_weared_incorrect:'Đeo sai',uncertain:'Chưa chắc chắn'};
const colors = {with_mask:'#16b394',without_mask:'#f36e72',mask_weared_incorrect:'#efb63f',uncertain:'#a2b4bd'};
const state = {base:'',stream:null,running:false,busy:false,connected:false,samples:[],warmup:true,model:null,generation:0,controller:null};
const canvas=$('result'),ctx=canvas.getContext('2d'),video=$('video');
$('endpoint').value=localStorage.getItem('masklab-endpoint')||'';
function notice(message,error=false){$('notice').textContent=message;$('notice').classList.toggle('error',error);}
function baseUrl(){const value=$('endpoint').value.trim().replace(/\/+$/,'');if(!value)return location.origin;const u=new URL(value);if(!['http:','https:'].includes(u.protocol))throw Error('Backend phải dùng http hoặc https.');if(location.protocol==='https:'&&u.protocol==='http:'&&!['localhost','127.0.0.1'].includes(u.hostname))throw Error('Trang HTTPS cần backend HTTPS.');return u.origin+u.pathname.replace(/\/+$/,'');}
async function connect(){
  if(state.busy||state.running){notice('Dừng camera và đợi xử lý xong trước khi đổi backend.',true);return;}
  $('connect').disabled=true;
  notice('Đang kết nối backend… Dịch vụ cloud miễn phí có thể cần khoảng một phút để khởi động.');
  try{
    const base=baseUrl();const r=await fetch(base+'/api/health',{signal:AbortSignal.timeout(120000)});
    if(!r.ok)throw Error(`Backend chưa sẵn sàng (${r.status}).`);
    const health=await r.json();if(state.base!==base||state.model?.model_sha256!==health.model_sha256){state.samples=[];stats();}state.base=base;state.connected=true;state.warmup=true;state.model=health;
    localStorage.setItem('masklab-endpoint',$('endpoint').value.trim());
    $('connection').className='connection ok';$('connection').replaceChildren();const dot=document.createElement('i');$('connection').append(dot,document.createTextNode('Đã kết nối backend'));
    const host=new URL(base).hostname;const local=['localhost','127.0.0.1','::1','[::1]'].includes(host);
    $('mode').textContent=local?'Đang đo local ↔ local':`Backend từ xa: ${host}`;
    notice(local?'Backend local sẵn sàng. Nhập URL cloud để đo đúng local–cloud.':'Backend từ xa đã kết nối. Bạn có thể bật camera hoặc chọn ảnh.');
    $('postgres').disabled=!health.database_configured;
    if(!health.database_configured)$('postgres').checked=false;
    try{const info=await fetch(base+'/api/model-info').then(r=>r.json());const report=info.evaluation?.classification_report;
      $('model-details').textContent=`${info.architecture}. Thiết bị train: ${info.evaluation?.device||'—'}. Test accuracy: ${report?(100*report.accuracy).toFixed(2)+'%':'—'}. Macro F1: ${report?report['macro avg']['f1-score'].toFixed(3):'—'}. Đánh giá trên ảnh cắt khuôn mặt; khả năng phát hiện camera cần kiểm tra riêng. SHA256: ${health.model_sha256.slice(0,16)}…`;
    }catch{$('model-details').textContent='Không tải được đánh giá model.';}
  }catch(e){state.connected=false;$('connection').className='connection';$('connection').textContent='Kết nối thất bại';notice(e.message+' Nếu Render đang ngủ, hãy chờ khoảng một phút rồi kết nối lại.',true);}
  finally{$('connect').disabled=false;}
}
function stop(){state.running=false;state.generation++;state.controller?.abort();if(state.stream)state.stream.getTracks().forEach(t=>t.stop());state.stream=null;video.srcObject=null;$('start').disabled=false;$('stop').disabled=true;$('camera-select').disabled=false;$('source-status').textContent='Đã dừng';}
async function cameraOptions(){const devices=await navigator.mediaDevices.enumerateDevices();const chosen=$('camera-select').value;$('camera-select').replaceChildren();for(const d of devices.filter(d=>d.kind==='videoinput')){const option=document.createElement('option');option.value=d.deviceId;option.textContent=d.label||'Camera';$('camera-select').append(option);}if(chosen)$('camera-select').value=chosen;}
async function start(){
  if(state.busy){notice('Đợi ảnh hiện tại xử lý xong.',true);return;}
  if(!state.connected){await connect();if(!state.connected)return;}
  if(!navigator.mediaDevices?.getUserMedia){notice('Camera cần mở trang qua HTTPS hoặc localhost.',true);return;}
  try{
    const selected=$('camera-select').value;
    state.stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},...(selected?{deviceId:{exact:selected}}:{})},audio:false});
    video.srcObject=state.stream;await video.play();await cameraOptions();state.running=true;state.warmup=true;state.generation++;
    $('start').disabled=true;$('stop').disabled=false;$('camera-select').disabled=true;$('empty').hidden=true;$('source-status').textContent='Camera đang hoạt động';
    notice('Đang gửi từng khung hình đến backend. Bấm Dừng để ngừng camera và gửi ảnh.');loop(state.generation);
  }catch(e){stop();notice('Không mở được camera: '+e.message,true);}
}
function snapshot(source,width,height){const cap=document.createElement('canvas');const scale=Math.min(1,1280/width,720/height);cap.width=Math.round(width*scale);cap.height=Math.round(height*scale);cap.getContext('2d').drawImage(source,0,0,cap.width,cap.height);return cap;}
const blobOf=cap=>new Promise((resolve,reject)=>cap.toBlob(blob=>blob?resolve(blob):reject(Error('Không mã hóa được ảnh.')),'image/jpeg',.88));
const nextPaint=()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
function render(cap,result){
  canvas.width=cap.width;canvas.height=cap.height;ctx.drawImage(cap,0,0);
  const font=Math.max(12,Math.round(cap.width/75));ctx.font=`600 ${font}px sans-serif`;ctx.lineWidth=Math.max(2,cap.width/500);
  for(const d of result.detections){const [x1,y1,x2,y2]=d.box;const color=colors[d.label];ctx.strokeStyle=color;ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    const text=`${names[d.label]} · ${(d.confidence*100).toFixed(0)}%`;const tw=ctx.measureText(text).width+12;const top=Math.max(0,y1-font-12);const left=Math.min(x1,Math.max(0,cap.width-tw));ctx.fillStyle=color;ctx.fillRect(left,top,tw,font+9);ctx.fillStyle='#0b2228';ctx.fillText(text,left+6,top+font+1);
  }
  const keys=['correct','absent','incorrect','uncertain'];labels.forEach((label,i)=>$('count-'+keys[i]).textContent=result.counts[label]||0);
  $('face-total').textContent=result.face_count+' khuôn mặt';$('detected-list').replaceChildren();
  result.detections.forEach((d,i)=>{const row=document.createElement('div');row.className='face-row';const no=document.createElement('span');no.textContent='#'+(i+1);const text=document.createElement('strong');text.textContent=names[d.label];text.style.color=colors[d.label];const confidence=document.createElement('b');confidence.textContent=(d.confidence*100).toFixed(1)+'%';row.append(no,text,confidence);$('detected-list').append(row);});
  if(!result.face_count){const p=document.createElement('p');p.textContent='Không tìm thấy khuôn mặt. Thử đứng gần camera, đủ sáng và nhìn thẳng.';$('detected-list').append(p);}
}
function stats(){const rows=state.samples.filter(r=>!r.warmup);$('samples').textContent=rows.length+' mẫu ổn định';$('export').disabled=!state.samples.length;if(!rows.length){$('mean').textContent=$('p95').textContent='—';$('fps').textContent='— lượt/s';return;}const vals=rows.map(r=>r.end_to_end_ms).sort((a,b)=>a-b);const avg=vals.reduce((a,b)=>a+b,0)/vals.length;$('mean').textContent=avg.toFixed(1);$('p95').textContent=vals[Math.ceil(.95*vals.length)-1].toFixed(1);$('fps').textContent=(1000/avg).toFixed(1)+' lượt/s suy ra từ toàn trình';}
async function infer(cap,t0,generation){
  state.busy=true;$('processing').hidden=false;$('upload').disabled=true;
  try{
    const blob=await blobOf(cap);const data=new FormData();data.append('file',blob,'frame.jpg');const threshold=$('threshold').value/100;
    state.controller=new AbortController();const timeout=setTimeout(()=>state.controller?.abort(),120000);
    let result,send,received;
    try{send=performance.now();const response=await fetch(state.base+'/api/predict?classification_threshold='+threshold,{method:'POST',body:data,signal:state.controller.signal});result=await response.json();received=performance.now();if(!response.ok)throw Error(typeof result.detail==='string'?result.detail:`Lỗi API ${response.status}`);}
    finally{clearTimeout(timeout);state.controller=null;}
    if(generation!==state.generation)return;
    render(cap,result);$('empty').hidden=true;await nextPaint();
    const end=performance.now(),rtt=received-send,e2e=end-t0;
    $('inference').textContent=result.timing.inference_ms.toFixed(1);$('rtt').textContent=rtt.toFixed(1);$('e2e').textContent=e2e.toFixed(1);
    const row={timestamp:new Date().toISOString(),request_id:result.request_id,backend:state.base,model_sha256:result.model_sha256,
      warmup:state.warmup,inference_ms:result.timing.inference_ms,face_detection_ms:result.timing.face_detection_ms,cnn_ms:result.timing.cnn_ms,
      server_processing_ms:result.timing.server_processing_ms,queue_ms:result.timing.queue_ms,round_trip_ms:rtt,end_to_end_ms:e2e,
      local_encode_and_render_ms:e2e-rtt,transport_and_http_overhead_ms:Math.max(0,rtt-result.timing.server_processing_ms),
      classification_threshold:threshold,detection_threshold:.7,face_count:result.face_count,...result.counts};
    state.samples.push(row);if(state.samples.length>10000)state.samples.shift();state.warmup=false;stats();
    if($('postgres').checked){fetch(state.base+'/api/measurements',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({request_id:result.request_id,model_sha256:result.model_sha256,inference_ms:row.inference_ms,round_trip_ms:rtt,end_to_end_ms:e2e,counts:result.counts})}).catch(()=>notice('Nhận dạng thành công; chưa gửi được số đo PostgreSQL.',true));}
    if(!state.running)notice(`Đã nhận dạng ${result.face_count} khuôn mặt. ${row.warmup?'Mẫu warm-up được tách khỏi thống kê ổn định.':''}`);
  }catch(e){if(e.name==='AbortError'){if(generation===state.generation){notice('Yêu cầu đã quá thời gian. Hãy kiểm tra backend và kết nối lại.',true);if(state.running)stop();}}else{notice('Nhận dạng thất bại: '+e.message,true);if(state.running)stop();}}
  finally{state.busy=false;$('processing').hidden=true;$('upload').disabled=false;}
}
async function loop(generation){while(state.running&&generation===state.generation){if(!state.busy&&video.videoWidth){const t0=performance.now();const cap=snapshot(video,video.videoWidth,video.videoHeight);await infer(cap,t0,generation);}await new Promise(r=>setTimeout(r,100));}}
async function upload(event){const file=event.target.files[0];if(!file)return;stop();if(state.busy){notice('Đợi yêu cầu hiện tại dừng rồi chọn lại ảnh.',true);event.target.value='';return;}if(!state.connected){await connect();if(!state.connected)return;}let bitmap;try{bitmap=await createImageBitmap(file,{imageOrientation:'from-image'});const t0=performance.now();const cap=snapshot(bitmap,bitmap.width,bitmap.height);$('source-status').textContent=file.name;await infer(cap,t0,state.generation);}catch(e){notice('Không đọc được ảnh: '+e.message,true);}finally{bitmap?.close();event.target.value='';}}
function exportCsv(){if(!state.samples.length)return;const keys=Object.keys(state.samples[0]);const escape=v=>'"'+String(v??'').replaceAll('"','""')+'"';const csv='\ufeff'+[keys.join(','),...state.samples.map(r=>keys.map(k=>escape(r[k])).join(','))].join('\r\n');const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='masklab-latency-'+new Date().toISOString().replace(/[:.]/g,'-')+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
async function sample(){stop();if(state.busy){notice('Đợi yêu cầu hiện tại dừng rồi thử lại.',true);return;}if(!state.connected){await connect();if(!state.connected)return;}try{const response=await fetch('/static/samples/three_classes.jpg');if(!response.ok)throw Error('Chưa có ảnh mẫu đã kiểm tra. Hãy chọn ảnh của bạn.');const blob=await response.blob();await upload({target:{files:[new File([blob],'Ảnh mẫu ba lớp.jpg',{type:'image/jpeg'})],value:''}});}catch(e){notice(e.message,true);}}
async function benchmark(){
  stop();if(state.busy){notice('Đợi yêu cầu hiện tại dừng rồi đo lại.',true);return;}
  if(!state.connected){await connect();if(!state.connected)return;}
  $('benchmark').disabled=true;let bitmap;
  try{
    const response=await fetch('/static/samples/three_classes.jpg');if(!response.ok)throw Error('Chưa có ảnh mẫu để đo.');
    bitmap=await createImageBitmap(await response.blob());state.samples=[];stats();state.warmup=true;state.running=true;
    const generation=++state.generation;$('start').disabled=true;$('stop').disabled=false;$('camera-select').disabled=true;
    for(let i=0;i<31&&state.running&&generation===state.generation;i++){
      $('source-status').textContent=i===0?'Đo warm-up':`Đang đo mẫu ${i}/30`;
      const t0=performance.now();const cap=snapshot(bitmap,bitmap.width,bitmap.height);await infer(cap,t0,generation);
      await new Promise(resolve=>setTimeout(resolve,100));
    }
    const completed=state.running&&generation===state.generation;stop();
    if(completed)notice('Đã đo 30 mẫu ổn định và 1 warm-up trên ảnh mẫu. Bấm Xuất CSV để lưu đầy đủ ba độ trễ.');
  }catch(e){stop();notice('Không hoàn tất phép đo: '+e.message,true);}
  finally{bitmap?.close();$('benchmark').disabled=false;}
}
$('connect').addEventListener('click',connect);$('start').addEventListener('click',start);$('stop').addEventListener('click',()=>{stop();notice('Đã tắt camera. Kết quả cuối được giữ lại.');});$('upload').addEventListener('change',upload);$('export').addEventListener('click',exportCsv);$('clear').addEventListener('click',()=>{state.samples=[];state.warmup=true;stats();});$('threshold').addEventListener('input',()=>$('threshold-value').textContent=$('threshold').value+'%');window.addEventListener('pagehide',stop);
connect();
$('sample').addEventListener('click',sample);
$('benchmark').addEventListener('click',benchmark);
