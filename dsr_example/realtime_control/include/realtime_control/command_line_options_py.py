#!/usr/bin/env python
import argparse
from dataclasses import dataclass
from enum import Enum
import sched
import sys
from typing import Optional

class SchedPolicy(Enum):
    """Scheduling policy options"""
    SCHED_OTHER = sched.SCHED_OTHER if hasattr(sched, 'SCHED_OTHER') else 0
    SCHED_FIFO = sched.SCHED_FIFO if hasattr(sched, 'SCHED_FIFO') else 1
    SCHED_RR = sched.SCHED_RR if hasattr(sched, 'SCHED_RR') else 2

@dataclass
class SchedOptions:
    """Scheduling options data class"""
    priority: int = 0
    policy: SchedPolicy = SchedPolicy.SCHED_OTHER

class SchedOptionsReader:
    """Reader for scheduling command line options"""
    OPTION_PRIORITY = "--priority"
    OPTION_SCHED = "--sched"

    def __init__(self):
        self._options = SchedOptions()
        self._parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            description='Real-time scheduling options',
            formatter_class=argparse.RawTextHelpFormatter
        )
        
        parser.add_argument(
            self.OPTION_PRIORITY,
            type=int,
            default=0,
            help='Set real-time scheduling priority (min 1, max 99). Defaults to 0.'
        )
        
        parser.add_argument(
            self.OPTION_SCHED,
            type=str,
            choices=['SCHED_OTHER', 'SCHED_FIFO', 'SCHED_RR'],
            default='SCHED_OTHER',
            help='Set scheduling policy. Defaults to SCHED_OTHER.'
        )
        
        return parser

    def print_usage(self) -> None:
        """Print usage information"""
        self._parser.print_help()

    def get_options(self) -> SchedOptions:
        """Get the current options"""
        return self._options

    def read_options(self, args: Optional[list] = None) -> bool:
        """
        Read and validate command line options
        
        Args:
            args: Command line arguments (optional, uses sys.argv if None)
            
        Returns:
            bool: True if options were successfully read and validated
        """
        try:
            if args is None:
                args = sys.argv[1:]
            
            parsed_args = self._parser.parse_args(args)
            
            # Convert string policy to enum
            policy = getattr(SchedPolicy, parsed_args.sched)
            priority = parsed_args.priority

            # Validate options
            if policy in (SchedPolicy.SCHED_FIFO, SchedPolicy.SCHED_RR):
                if not 1 <= priority <= 99:
                    print("ERROR: With a real-time sched policy the priority has to be between 1 and 99")
                    return False
            elif policy == SchedPolicy.SCHED_OTHER:
                if priority != 0:
                    print("ERROR: Use a real-time scheduling policy to set a real-time priority")
                    return False

            self._options.priority = priority
            self._options.policy = policy
            return True

        except Exception as e:
            print(f"Error parsing arguments: {str(e)}")
            self.print_usage()
            return False

def main():
    """Example usage of SchedOptionsReader"""
    reader = SchedOptionsReader()
    if not reader.read_options():
        reader.print_usage()
        sys.exit(1)

    options = reader.get_options()
    print(f"Selected options: priority={options.priority}, policy={options.policy.name}")

if __name__ == "__main__":
    main()
