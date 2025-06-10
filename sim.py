import os, csv
import simpy
import numpy as np
import random
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from pred_helpers import predict_violation


# Constants
SIM_TIME = 1000
TOTAL_BW = 17 # Mbps

# Traffic types
SLICES = {
    "eMBB":{"rate": 12, "priority": 1,"jitter" :6,"minSLA":0},

    "URLLC":{"rate": 4.5, "priority": 2,"jitter" : 1,"minSLA": 3.5},
    "mMTC":{"rate": 0.4, "priority": 3, "jitter" : 0.2, "minSLA": 0.3 }
}

true_v=[]
pred_v=[]
class BandwidthTracker:
    def __init__(self):
        self.total_usage_ask_log=[]
        self.total_admitted_log=[]
        self.total_dropped_log=[]
        self.slice_usage_ask_log={name: [] for name in SLICES}
        self.slice_usage_allocated_log={name: [] for name in SLICES}
        self.slice_dropped_log = {name: [] for name in SLICES}

        #for prediction vs actual sla violation comparisons.
        self.slice_pred_log = {name: [] for name in SLICES}
        self.slice_true_log = {name: [] for name in SLICES}


    def log_usage_ask(self, time, usage_by_slice):
        total = sum(usage_by_slice.values())
        self.total_usage_ask_log.append((time,total))
        for name, bw in usage_by_slice.items():
            self.slice_usage_ask_log[name].append((time,bw))

    def log_admitted(self,time,admit):
        total=sum(admit.values())
        self.total_admitted_log.append((time, total))

        for name,bw in admit.items(): 
             self.slice_usage_allocated_log[name].append((time,bw))

    def log_dropped(self,time,drop):
        total=sum(drop.values())
        self.total_dropped_log.append((time, total))

        for name in SLICES:
            bw= drop.get(name,0)
            self.slice_dropped_log[name].append((time,bw))


class SliceTraffic:
    def __init__(self, env, name, rate, jitter, priority, tracker,minSLA):
        self.env = env
        self.name = name
        self.rate = rate
        self.jitter = jitter
        self.priority = priority
        self.tracker = tracker
        self.minSLA= minSLA
       # self.usage_log = usage_log
        self.process = env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(1)
            usage = self.rate + random.uniform(-0.2, 0.2) * self.jitter
            
            self.tracker.slice_usage_ask_log[self.name].append((self.env.now, max(0,usage)))
            

def simulate():
    env = simpy.Environment()
    tracker = BandwidthTracker()


    slices = [
        SliceTraffic(env, name, SLICES[name]["rate"], SLICES[name]["jitter"], SLICES[name]["priority"],tracker, SLICES[name]["minSLA"])
        for name in SLICES
    ]

    def monitor_ask(env,tracker,sla_violations):
        urllc_priority= False
        prio_end = 0
        
        while True:
            yield env.timeout(1)

            prob=random.uniform(0,1)
            if not urllc_priority:
                if prob>0.995:
                    urllc_priority= True
                    prio_start=env.now
                    prio_end=env.now+75
                    SLICES["URLLC"]["priority"]=0
                    print(f"prio raised at{env.now}")
            if urllc_priority and env.now>prio_end:
                urllc_priority = False
                SLICES["URLLC"]["priority"]=2

            
            current_ask = {
                name: tracker.slice_usage_ask_log[name][-1][1] if tracker.slice_usage_ask_log[name] else 0
                for name in SLICES
            }
            tracker.log_usage_ask(env.now, current_ask)

            #sorting slice:
            sorted_slices=sorted(SLICES.items(),key=lambda x:x[1] ["priority"])

            admitted={}
            dropped={}
            remaining_bw=TOTAL_BW#*random.triangular(0.8,1,0.85)
            #uncommet the previous line extension if needed. Not entirely confident but just trying something out to mimic real world bandwidth fluctuations.
            remaining_bw-=remaining_bw*random.uniform(0.005,0.025)
        
            #above is just an overhead bw consideration

            for name,meta in sorted_slices:
                req=current_ask[name]
                if req<=remaining_bw:
                    admitted[name]=req
                    remaining_bw-=req
                else:
                    admitted[name]=remaining_bw
                    dropped[name]=max(0,req-remaining_bw)
                    remaining_bw=0

                if admitted.get(name,0)< SLICES[name]["minSLA"]:
                    sla_violations[name]+=1

                #logging slice violation
                true_violation = int(admitted.get(name,0)<SLICES[name]["minSLA"])
                tracker.slice_true_log[name].append((env.now,true_violation))
                
                #logging predicted violation
                rt_pred=predict_violation(name,current_ask[name],admitted[name],dropped.get(name,0))
                tracker.slice_pred_log[name].append((env.now,rt_pred))
                #print(f"[t={env.now:4}] {name:<6}  RF-pred = {rt_pred}")

            #print(f"\nTime {env.now}")
            #print(f"Ask:     {current_ask}")
            #print(f"Admitted:{admitted}")
            #print(f"Dropped: {dropped}")



            tracker.log_admitted(env.now, admitted)
            tracker.log_dropped(env.now,dropped)
    
    
    sla_violations={name:0 for name in SLICES}
    
    env.process(monitor_ask(env, tracker,sla_violations))

    env.run(until=SIM_TIME)
    for name in SLICES:
        true_slice=[v for _,v in tracker.slice_true_log[name]]
        pred_slice=[v for _,v in tracker.slice_pred_log[name]]

        true_v.extend(true_slice)
        pred_v.extend(pred_slice)

    print("/////True violations vs Predicted Violations metrics://///")
    print(confusion_matrix(true_v,pred_v))
    print(classification_report(true_v,pred_v,digits=3))
    
    
    print("\n====== SLA Violation Summary ======")
    for name in SLICES:
        print(f"{name}: {sla_violations[name]} violations out of {SIM_TIME} ticks")

    logs_to_csv(tracker, sla_violations)

    return tracker

def logs_to_csv(tracker,sla_violations,filename="logs/simulation_log.csv"):
    os.makedirs("logs",exist_ok=True)
    with open(filename,mode="w",newline='') as file:
        writer = csv.writer(file)

        #Header:
        writer.writerow([
            "Time","Slice","Asked","Admitted","Dropped","SLA_violation"
        ])

        for t in range(SIM_TIME):
            for name in SLICES:
                ask=get_bw_at_time(tracker.slice_usage_ask_log[name],t)
                adm=get_bw_at_time(tracker.slice_usage_allocated_log[name],t)
                drp=get_bw_at_time(tracker.slice_dropped_log[name],t)
                if adm < SLICES[name]["minSLA"]:
                    violation=1    
                else:
                    violation=0
                writer.writerow([t,name,ask,adm,drp,violation])

def get_bw_at_time(log,t):
    for time, bw in log:
        if time==t:
            return bw
    return 0


def plot_usage(tracker):
    plt.figure(figsize=(10, 6))
    for name in SLICES:
        time,ask_bw=zip(*tracker.slice_usage_ask_log[name])
        time_adm,adm_bw=zip(*tracker.slice_usage_allocated_log[name])
        
        if tracker.slice_dropped_log[name]:
            time_dr,drop_bw=zip(*tracker.slice_dropped_log[name])
            plt.plot(time_dr,drop_bw, label=f"{name}- Dropped")

    
        plt.plot(time,ask_bw, label=f"{name}- ASked")
        plt.plot(time_adm,adm_bw, label=f"{name}- Admitted")
        
        
    
    
    
    times, totals = zip(*tracker.total_usage_ask_log)
    plt.plot(times, totals, label="Total", linestyle="--", linewidth=2, color="black")
    plt.axhline(y=TOTAL_BW, color='red', linestyle=':', label="Capacity Limit")
    plt.xlabel("Time (s)")
    plt.ylabel("Bandwidth Usage (Mbps)")
    plt.title("Network Slice + Total Bandwidth Usage Over Time")
    plt.legend()
    #plt.grid(True)
    plt.tight_layout()
    #plt.savefig("network_usage_final.png")
    plt.show()

if __name__ == "__main__":
    tracker = simulate()


    #plot_usage(tracker)