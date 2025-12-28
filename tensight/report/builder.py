from datetime import datetime
from typing import List, Dict, Any

class Problem:
    
    
    SEVERITY_ERROR = "error"
    SEVERITY_WARNING = "warning"
    SEVERITY_INFO = "info"
    
    def __init__(
        self,
        name: str,
        severity: str,
        description: str,
        suggestion: str,
        details: Dict[str, Any] = None,
        paper_ref: str = None
    ):
        self.name = name
        self.severity = severity
        self.description = description
        self.suggestion = suggestion
        self.details = details or {}
        self.paper_ref = paper_ref
        self.timestamp = datetime.now()
    
    def __repr__(self):
        return f"Problem({self.name}, {self.severity})"


class Report:
    
    
    def __init__(self, model_name: str = "model"):
        self.model_name = model_name
        self.problems: List[Problem] = []
        self.good_things: List[str] = []
        self.stats: Dict[str, Any] = {}
        self.analyses: Dict[str, Any] = {}
        self.timestamp = datetime.now()
    
    def add_problem(self, problem: Problem):
        self.problems.append(problem)
    
    def add_problems(self, problems: List[Problem]):
        self.problems.extend(problems)
    
    def add_good(self, message: str):
        self.good_things.append(message)
    
    def add_stat(self, key: str, value: Any):
        self.stats[key] = value
    
    def add_analysis(self, name: str, results: Dict[str, Any]):
        self.analyses[name] = results
    
    @property
    def error_count(self) -> int:
        return len([p for p in self.problems if p.severity == Problem.SEVERITY_ERROR])
    
    @property
    def warning_count(self) -> int:
        return len([p for p in self.problems if p.severity == Problem.SEVERITY_WARNING])
    
    @property
    def health_score(self) -> str:
        if self.error_count > 0:
            return "🔴 CRITICAL"
        elif self.warning_count > 2:
            return "🟡 RISKY"
        elif self.warning_count > 0:
            return "🟢 OK (with warnings)"
        else:
            return "✅ PERFECT"
    
    @property
    def can_train(self) -> bool:
        return self.error_count == 0
    
    def display(self):
        
        
        print("\n")
        print("╔══════════════════════════════════════════════════════════╗")
        print("║           🔍 TENSIGHT DIAGNOSTIC REPORT                  ║")
        print("╠══════════════════════════════════════════════════════════╣")
        print(f"║  Model: {self.model_name:<47} ║")
        print(f"║  Time:  {self.timestamp.strftime('%Y-%m-%d %H:%M:%S'):<47} ║")
        print("╚══════════════════════════════════════════════════════════╝")
        
        
        print(f"\n🏥 Health Score: {self.health_score}")
        if self.error_count > 0:
            print("📣 Recommendation: DO NOT START TRAINING")
        elif self.warning_count > 0:
            print("📣 Recommendation: Fix warnings first")
        else:
            print("📣 Recommendation: Good to go! 🚀")
        
        
        if self.stats:
            print("\n📊 Statistics:")
            print("-" * 40)
            for key, value in self.stats.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                elif isinstance(value, int):
                    print(f"   {key}: {value:,}")
                else:
                    print(f"   {key}: {value}")
        
        
        if self.problems:
            print(f"\n⚠️ Problems Detected: {len(self.problems)}")
            print("-" * 40)
            
            sorted_problems = sorted(
                self.problems,
                key=lambda p: 0 if p.severity == "error" else 1
            )
            
            for p in sorted_problems:
                icon = "🔴" if p.severity == "error" else "🟡"
                print(f"\n{icon} {p.name}")
                print(f"   {p.description}")
                print(f"   💡 {p.suggestion}")
                if p.paper_ref:
                    print(f"   📄 Ref: {p.paper_ref}")
        else:
            print("\n✅ No problems detected!")
        
        
        if self.good_things:
            print("\n✅ What's Good:")
            for good in self.good_things:
                print(f"   • {good}")
        
        
        if self.analyses:
            print("\n🔬 Advanced Analyses:")
            print("-" * 40)
            for name, results in self.analyses.items():
                print(f"\n   📈 {name}:")
                for key, value in results.items():
                    if isinstance(value, float):
                        print(f"      {key}: {value:.4f}")
                    else:
                        print(f"      {key}: {value}")
        
        print("\n" + "═" * 60)
        print("🔍 Tensight - See through your models")
        print("═" * 60 + "\n")
    
    def to_dict(self) -> Dict[str, Any]:
        
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat(),
            "health_score": self.health_score,
            "can_train": self.can_train,
            "stats": self.stats,
            "problems": [
                {
                    "name": p.name,
                    "severity": p.severity,
                    "description": p.description,
                    "suggestion": p.suggestion,
                }
                for p in self.problems
            ],
            "analyses": self.analyses,
        }