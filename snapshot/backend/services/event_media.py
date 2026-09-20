"""Delete one event and its own media, with strict directory ownership checks."""
from pathlib import Path
import re
import time

from ..database.db import loads


class MediaDeleteError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def delete_event_with_media(repo, events_dir: Path, event_id: str):
    """Delete media first, then DB records; interrupted calls can be retried.

    All directories and references are validated before the first unlink. On a
    filesystem failure the event remains in the database for retry; files already
    removed during that attempt may be missing. The DB lock serializes deletes.
    """
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', event_id):
        raise MediaDeleteError(404, '事件不存在')
    root = Path(events_dir).resolve()
    with repo.db.cursor() as cur:
        row = cur.execute('SELECT * FROM risk_events WHERE event_id=?', (event_id,)).fetchone()
        if row is None:
            return False  # Idempotent retry after a lost success response.
        basis = loads(row['rule_basis'], {})
        visual = basis.get('visual') or {}
        if visual.get('state') == 'pending_review':
            raise MediaDeleteError(409, '事件正在保存或复核，请完成后再删除')
        if row['status'] == 'PENDING' and visual.get('alarm_held'):
            raise MediaDeleteError(409, '此事件仍维持报警，请先完成事件处理再删除')
        if time.time() - row['created_at'] < 5:
            raise MediaDeleteError(409, '事件刚刚生成，请等待证据保存完成后再删除')
        paths = [row['raw_image'], row['annotated_image'], visual.get('evidence_clip'),
                 (basis.get('audio') or {}).get('path')]
        for frame in cur.execute('SELECT raw_path, annotated_path FROM event_frames WHERE event_id=?', (event_id,)):
            paths.extend([frame['raw_path'], frame['annotated_path']])
        folders = set()
        for relative in filter(None, paths):
            path = Path(relative)
            target = (root / path).resolve()
            if (path.is_absolute() or len(path.parts) != 3 or not re.fullmatch(r'\d{8}', path.parts[0])
                    or path.parts[1] != event_id or target.parent != root / path.parts[0] / event_id
                    or not target.is_relative_to(root)):
                raise MediaDeleteError(409, '事件附件路径异常，未删除任何文件')
            folders.add(root / path.parts[0] / event_id)
        # Also remove orphan media left in this event's directory by interrupted
        # evidence generation, including folders without a remaining DB reference.
        if root.exists():
            for day in root.iterdir():
                if re.fullmatch(r'\d{8}', day.name) and (day / event_id).exists():
                    folders.add(day / event_id)
        files, directories = [], []
        for folder in folders:
            if (folder.is_symlink() or folder.is_junction() or folder.parent.is_symlink()
                    or folder.parent.is_junction() or folder.resolve() != folder
                    or not folder.is_relative_to(root)):
                raise MediaDeleteError(409, '事件附件目录异常，未删除任何文件')
            if not folder.exists():
                continue
            if not folder.is_dir():
                raise MediaDeleteError(409, '事件附件目录异常，未删除任何文件')
            entries = [folder]
            while entries:
                entry = entries.pop()
                if entry.is_symlink() or entry.is_junction() or not entry.resolve().is_relative_to(folder):
                    raise MediaDeleteError(409, '事件附件含外部链接，未删除任何文件')
                if entry.is_dir():
                    directories.append(entry)
                    entries.extend(entry.iterdir())
                else:
                    files.append(entry)
        try:
            for target in files:
                target.unlink(missing_ok=True)
            for folder in sorted(set(directories), key=lambda p: len(p.parts), reverse=True):
                folder.rmdir()
        except PermissionError as exc:
            raise MediaDeleteError(409, '附件被占用或不可写，事件记录仍保留；请关闭播放器后重试删除') from exc
        except OSError as exc:
            raise MediaDeleteError(500, '部分附件未能删除，事件记录仍保留；请检查存储后重试') from exc
        cur.execute('DELETE FROM alert_actions WHERE event_id=?', (event_id,))
        cur.execute('DELETE FROM operator_actions WHERE event_id=?', (event_id,))
        # event_frames and gpt_analyses have ON DELETE CASCADE foreign keys.
        cur.execute('DELETE FROM risk_events WHERE event_id=?', (event_id,))
        cur.execute('INSERT INTO operator_actions (event_id, action, operator, note, created_at) '
                    'VALUES (?, ?, ?, ?, ?)',
                    (None, 'delete_event', 'local', f'已删除事件及全部附件：{event_id}', time.time()))
    return True
