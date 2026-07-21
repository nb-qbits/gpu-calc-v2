import styles from './ComingSoonRibbon.module.css'

export default function ComingSoonRibbon({ children }: { children: React.ReactNode }) {
  return (
    <div className={styles.wrapper}>
      <div className={styles.ribbon}>Coming soon</div>
      {children}
    </div>
  )
}
