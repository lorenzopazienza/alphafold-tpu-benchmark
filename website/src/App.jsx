import { lazy, Suspense } from 'react'
import Nav from './components/Nav'
import Hero from './components/Hero'
import Problem from './components/Problem'
import Architecture from './components/Architecture'
import HeadlineResults from './components/HeadlineResults'
import DeepDive from './components/DeepDive'
import CostTakeaways from './components/CostTakeaways'
import Repo from './components/Repo'
import Footer from './components/Footer'

const ProteinViewer = lazy(() => import('./components/ProteinViewer'))

export default function App() {
  return (
    <div className="min-h-svh bg-paper text-ink">
      <Nav />
      <main>
        <Hero />
        <Suspense
          fallback={
            <section
              id="structure"
              className="border-t border-line bg-panel"
              aria-busy="true"
            >
              <div className="shell py-24 text-sm text-mute">
                Loading structure…
              </div>
            </section>
          }
        >
          <ProteinViewer />
        </Suspense>
        <Problem />
        <Architecture />
        <HeadlineResults />
        <DeepDive />
        <CostTakeaways />
        <Repo />
      </main>
      <Footer />
    </div>
  )
}
