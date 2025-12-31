"""
Report builder for Tensight diagnostics.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional


class Problem:
    """Represents a detected problem."""
    
    SEVERITY_ERROR = "error"
    SEVERITY_WARNING = "warning"
    SEVERITY_INFO = "info"
    
    def __init__(
        self,
        name: str,
        severity: str,
        description: str,
        suggestion: str,
        details: Optional[Dict[str, Any]] = None,
        paper_ref: Optional[str] = None
    ):
        """
        Create a new Problem.
        
        Args:
            name: Short problem name
            severity: One of 'error', 'warning', 'info'
            description: What's wrong
            suggestion: How to fix it
            details: Additional data (optional)
            paper_ref: Research paper reference (optional)
        """
        self.name = name
        self.severity = severity
        self.description = description
        self.suggestion = suggestion
        self.details = details or {}
        self.paper_ref = paper_ref
        self.timestamp = datetime.now()
    
    def __repr__(self) -> str:
        return f"Problem({self.name!r}, severity={self.severity!r})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "severity": self.severity,
            "description": self.description,
            "suggestion": self.suggestion,
            "details": self.details,
            "paper_ref": self.paper_ref,
        }


class Report:
    """Diagnostic report containing problems and statistics."""
    
    def __init__(self, model_name: str = "model"):
        """
        Create a new Report.
        
        Args:
            model_name: Name of the model being analyzed
        """
        self.model_name = model_name
        self.problems: List[Problem] = []
        self.good_things: List[str] = []
        self.stats: Dict[str, Any] = {}
        self.analyses: Dict[str, Any] = {}
        self.timestamp = datetime.now()
    
    def add_problem(self, problem: Problem) -> None:
        """Add a single problem."""
        self.problems.append(problem)
    
    def add_problems(self, problems: List[Problem]) -> None:
        """Add multiple problems."""
        self.problems.extend(problems)
    
    def add_good(self, message: str) -> None:
        """Add a positive observation."""
        self.good_things.append(message)
    
    def add_stat(self, key: str, value: Any) -> None:
        """Add a statistic."""
        self.stats[key] = value
    
    def add_analysis(self, name: str, results: Dict[str, Any]) -> None:
        """Add analysis results."""
        self.analyses[name] = results
    
    @property
    def error_count(self) -> int:
        """Count of error-level problems."""
        return sum(1 for p in self.problems if p.severity == Problem.SEVERITY_ERROR)
    
    @property
    def warning_count(self) -> int:
        """Count of warning-level problems."""
        return sum(1 for p in self.problems if p.severity == Problem.SEVERITY_WARNING)
    
    @property
    def health_score(self) -> str:
        """Overall health assessment."""
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
        """Whether training is recommended."""
        return self.error_count == 0
    
    def display(self) -> None:
        """Print the report to terminal."""
        self._print_header()
        self._print_health()
        self._print_stats()
        self._print_problems()
        self._print_good_things()
        self._print_analyses()
        self._print_footer()
    
    def _print_header(self) -> None:
        """Print report header."""
        print("\n")
        print("╔" + "═" * 58 + "╗")
        print("║" + "🔍 TENSIGHT DIAGNOSTIC REPORT".center(58) + "║")
        print("╠" + "═" * 58 + "╣")
        print(f"║  Model: {self.model_name:<48}║")
        print(f"║  Time:  {self.timestamp.strftime('%Y-%m-%d %H:%M:%S'):<48}║")
        print("╚" + "═" * 58 + "╝")
    
    def _print_health(self) -> None:
        """Print health score."""
        print(f"\n🏥 Health Score: {self.health_score}")
        
        if self.error_count > 0:
            print("📣 Recommendation: DO NOT START TRAINING")
        elif self.warning_count > 0:
            print("📣 Recommendation: Fix warnings first")
        else:
            print("📣 Recommendation: Good to go! 🚀")
    
    def _print_stats(self) -> None:
        """Print statistics."""
        if not self.stats:
            return
        
        print("\n📊 Statistics:")
        print("-" * 45)
        for key, value in self.stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            elif isinstance(value, int):
                print(f"   {key}: {value:,}")
            else:
                print(f"   {key}: {value}")
    
    def _print_problems(self) -> None:
        """Print detected problems."""
        if not self.problems:
            print("\n✅ No problems detected!")
            return
        
        print(f"\n⚠️ Problems Detected: {len(self.problems)}")
        print("-" * 45)
        
        # Sort by severity (errors first)
        sorted_problems = sorted(
            self.problems,
            key=lambda p: (0 if p.severity == "error" else 
                          1 if p.severity == "warning" else 2)
        )
        
        for p in sorted_problems:
            icon = {
                "error": "🔴",
                "warning": "🟡",
                "info": "🔵"
            }.get(p.severity, "⚪")
            
            print(f"\n{icon} {p.name}")
            print(f"   {p.description}")
            print(f"   💡 {p.suggestion}")
            
            if p.paper_ref:
                print(f"   📄 Ref: {p.paper_ref}")
    
    def _print_good_things(self) -> None:
        """Print positive observations."""
        if not self.good_things:
            return
        
        print("\n✅ What's Good:")
        for item in self.good_things:
            print(f"   • {item}")
    
    def _print_analyses(self) -> None:
        """Print analysis results."""
        if not self.analyses:
            return
        
        print("\n🔬 Advanced Analyses:")
        print("-" * 45)
        
        for name, results in self.analyses.items():
            print(f"\n   📈 {name}:")
            for key, value in results.items():
                if isinstance(value, float):
                    print(f"      {key}: {value:.6f}")
                elif not isinstance(value, (list, dict, type(None))):
                    print(f"      {key}: {value}")
    
    def _print_footer(self) -> None:
        """Print report footer."""
        print("\n" + "═" * 60)
        print("🔍 Tensight - See through your models")
        print("═" * 60 + "\n")
    
    def to_dict(self) -> Dict[str, Any]:
        """Export report as dictionary."""
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat(),
            "health_score": self.health_score,
            "can_train": self.can_train,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "stats": self.stats,
            "problems": [p.to_dict() for p in self.problems],
            "good_things": self.good_things,
            "analyses": self.analyses,
        }
