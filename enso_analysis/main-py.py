"""
Command-line interface for ENSO Analysis.
"""

import os
import argparse
import logging
from datetime import datetime

from .oni import ONIAnalyzer
from .transition import TransitionAnalyzer
from .visualization import ENSOVisualizer


def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Set up logging configuration.
    
    Parameters
    ----------
    log_level : int, optional
        Logging level, defaults to logging.INFO
    log_file : str, optional
        Path to log file, defaults to None (console logging only)
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=log_level,
            format=log_format
        )
    
    logging.info("ENSO Analysis started")


def parse_args():
    """
    Parse command-line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='ENSO Analysis Toolkit')
    
    parser.add_argument(
        '--sst-file', type=str, required=True,
        help='Path to the SST NetCDF file'
    )
    
    parser.add_argument(
        '--output-dir', type=str, default='./output',
        help='Directory to save outputs (default: ./output)'
    )
    
    parser.add_argument(
        '--start-year', type=int, default=1940,
        help='Start year for analysis (default: 1940)'
    )
    
    parser.add_argument(
        '--end-year', type=int, default=2024,
        help='End year for analysis (default: 2024)'
    )
    
    parser.add_argument(
        '--log-level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--skip-transition', action='store_true',
        help='Skip transition analysis'
    )
    
    parser.add_argument(
        '--skip-visualization', action='store_true',
        help='Skip visualization'
    )
    
    return parser.parse_args()


def main():
    """Main function to run the ENSO analysis."""
    args = parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f'enso_analysis_{timestamp}.log')
    setup_logging(log_level, log_file)
    
    # Log arguments
    logging.info(f"Arguments: {args}")
    
    try:
        # ONI Analysis
        logging.info("Starting ONI analysis")
        oni_analyzer = ONIAnalyzer(args.sst_file, args.output_dir)
        oni_analyzer.load_sst_data()
        oni_analyzer.generate_climatology_periods(args.start_year, args.end_year)
        oni_analyzer.process_nino34_data()
        oni_analyzer.compute_full_oni()
        logging.info(f"ONI analysis complete. Data saved to {args.output_dir}")
        
        # Get event summary
        event_summary = oni_analyzer.get_event_summary()
        logging.info(f"Found {len(event_summary)} ENSO events")
        
        # Transition Analysis
        if not args.skip_transition:
            logging.info("Starting transition analysis")
            oni_file = os.path.join(args.output_dir, 'oni_data.csv')
            
            transition_analyzer = TransitionAnalyzer(args.sst_file, oni_file, args.output_dir)
            transition_analyzer.run_transition_analysis()
            logging.info(f"Transition analysis complete. Data saved to {args.output_dir}")
        
        # Visualization
        if not args.skip_visualization:
            logging.info("Creating visualizations")
            visualizer = ENSOVisualizer(args.output_dir)
            
            # ONI phases plot
            visualizer.plot_oni_phases(oni_analyzer.oni_df, oni_analyzer.elnino_thresholds)
            
            # Transition composites plot
            if not args.skip_transition:
                visualizer.plot_enso_transition_composites(transition_analyzer)
            
            logging.info(f"Visualizations complete. Plots saved to {args.output_dir}")
        
        logging.info("ENSO Analysis completed successfully")
    
    except Exception as e:
        logging.error(f"Error in ENSO Analysis: {str(e)}", exc_info=True)
        raise
    

if __name__ == "__main__":
    main()
