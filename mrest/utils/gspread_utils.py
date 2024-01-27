from os import stat
import gspread
import numpy as np
import time

import socket

from typing import Any, List, Mapping


gc = None

class SheetsRecorder:

    # How many rows of resutls do we add for each eval run.
    ROWS_TO_ADD_PER_EVAL_RUN = ['seed', 'wandb', 'result_dir']

    @staticmethod
    def create_client():
        global gc
        if gc is None:
            gc = gspread.oauth()
        return gc

    def __init__(self, gc, sheet_name: str, worksheet_name: str, header_row: int = 2) -> None:
        self.gc = gc
        sheet = gc.open(sheet_name)
        if sheet is None:
            print(f"Unable to open sheet: {sheet_name}")
            return

        self.worksheet = sheet.worksheet(worksheet_name)
        self.header_row = header_row
        self.headers = self.worksheet.row_values(header_row)

        experiment_ids = self.get_all_experiment_ids(self.headers)
        if len(experiment_ids) == 0:
            current_experiment_id = 1
        else:
            current_experiment_id = max(experiment_ids) + 1
        self.current_experiment_id = current_experiment_id

        self.wandb_runpath_col = 3
    
    @property
    def row_count(self):
        return self.worksheet.row_count
    
    def number_of_rows_to_add_per_eval_run(self) -> int:
        """Get number of rows of info that should be added for each eval run."""
        return len(SheetsRecorder.ROWS_TO_ADD_PER_EVAL_RUN)
    
    def find_rows_for_wandb_runpath(self, run_path: str) -> List[int]:
        wandb_runpath_column = self.wandb_runpath_col
        all_run_paths = self.worksheet.col_values(wandb_runpath_column)
        rows = []
        for rp_idx, rp in enumerate(all_run_paths):
            if rp.strip() == run_path:
                # Since rows are indexed starting from 1
                rows.append(rp_idx + 1)
        return rows
    
    def find_info_rows_for_wandb_runpath(self, run_path: str, eval_row: int) -> List[int]:
        all_run_paths = self.worksheet.col_values(self.wandb_runpath_col)
        rows = []
        for rp_idx, rp in enumerate(all_run_paths):
            if rp.strip() == run_path:
                # We store the eval row right after the run_path
                row_value = int(all_run_paths[rp_idx + 1])
                if row_value == eval_row:
                    # Since rows are indexed starting from 1
                    rows.append(rp_idx + 1)
        return rows

    def update_cell_with_value(self, row_idx: int, col_idx: int, epoch: int, value: float):
        start_col = 5
        col_idx = start_col + col_idx
        status = self.worksheet.update_cell(row_idx, col_idx, f'epoch {epoch}')
        status = self.worksheet.update_cell(row_idx + 1, col_idx, f'{value:.2f}')
        return status
    
    def get_all_wandb_runpaths(self) -> Mapping[str, List]:
        wandb_runpath_column = self.wandb_runpath_col
        runpaths = self.worksheet.col_values(wandb_runpath_column)
        return {
            'row_index': [i + 1 for i in range(len(runpaths))],
            'run_path': runpaths,
        }

    def get_all_wandb_upload_status(self) -> List:
        return self.worksheet.col_values(5)

    def row_values(self, idx: int) -> List:
        return self.worksheet.row_values(idx)
    
    def column_values(self, idx: int) -> List:
        return self.worksheet.col_values(idx)

    def get_all_experiment_ids(self, headers):
        assert 'id' in headers
        assert headers.index('id') == 0
        all_experiment_ids = self.worksheet.col_values(1)
        all_experiment_id_ints = [int(exp_id) for exp_id in all_experiment_ids if exp_id.isnumeric()]
        print(all_experiment_id_ints)
        return all_experiment_id_ints

    def get_header_index_for_name(self, header_name):
        if header_name in self.headers:
            return self.headers.index(header_name)
        else:
            return -1
        
    def record_runpath_with_eval_row(self, runpath: str, eval_row: int):
        """Record runpath with the row we used in the eval sheet. 
        
        This sheet is used to store the info for these runs."""
        row_count = len(self.column_values(self.wandb_runpath_col)) + self.number_of_rows_to_add_per_eval_run()
        self.worksheet.update_cell(row_count, self.wandb_runpath_col, runpath)
        self.worksheet.update_cell(row_count + 1, self.wandb_runpath_col, eval_row) 
        return row_count
    
    def record_info_for_eval_run(self, row_idx: int, col_idx, seed: int, wandb_url: str, result_dir: str):
        seed_fmt = f'seed: {seed}, {socket.gethostname()}'
        self.worksheet.update_cell(row_idx, col_idx, seed_fmt)
        self.worksheet.update_cell(row_idx + 1, col_idx, wandb_url)
        self.worksheet.update_cell(row_idx + 2, col_idx, result_dir)

    def record_initial_experiment_data(self, seed, env_name, result_dir, notes='', **kwargs):
        values = []
        for header in self.headers:
            if header == 'id':
                values.append(self.current_experiment_id)
            elif header == 'Seed':
                values.append(seed)
            elif header == 'hostname':
                values.append(socket.gethostname())
            elif header == 'task':
                values.append(env_name)
            elif header == 'model_type':
                values.append(kwargs['model_type'])
            elif header == 'Start Time':
                time_str = time.strftime('%l:%M%p %Z on %b %d, %Y')
                values.append(time_str)
            elif header == 'Result dir':
                values.append(result_dir)
            elif header == 'Notes':
                values.append(notes)
            else:
                values.append('')

        print(f"Will add values to gsheet: {values}")
        gsheet_resp = self.worksheet.append_row(values)
        print(f"Did add values to gsheet (response): {gsheet_resp}")
        